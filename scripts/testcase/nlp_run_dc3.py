#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import importlib.util
import json
import pickle
import sys
import time
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "nlpopt" / "src"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    import torch
except ImportError:
    torch = None

from scripts.misc.nlp_dataset_cache import ensure_cached_dataset  # noqa: E402
from scripts.misc.runtime_seed import seed_torch_runtime  # noqa: E402
from scripts.misc.solver_config import resolve_solver_name  # noqa: E402
from scripts.misc.training_timing import should_track_epoch, summarize_timing_profile, timing_window_label  # noqa: E402


def _json_hash(payload: dict) -> str:
    return hashlib.sha1(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()[:10]


def _str_to_bool(value):
    if isinstance(value, bool):
        return value
    lowered = str(value).lower()
    if lowered in {"false", "f", "0", "no", "n"}:
        return False
    if lowered in {"true", "t", "1", "yes", "y"}:
        return True
    raise ValueError(f"{value!r} is not a valid boolean value")


def _load_json(path: Path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _validate_data_cfg(data_cfg: dict) -> None:
    n_x = int(data_cfg["n_x"])
    n_y = int(data_cfg["n_y"])
    n_eq = int(data_cfg["n_eq"])
    n_ineq = int(data_cfg["n_ineq"])
    n_samples = int(data_cfg["N_samples"])
    n_points = int(data_cfg["N_points"])
    x_l = np.asarray(data_cfg["x_L"], dtype=float)
    x_u = np.asarray(data_cfg["x_U"], dtype=float)

    if str(data_cfg["type"]).strip().lower() != "nlp":
        raise ValueError("Unsupported NLP testcase type.")
    if n_x <= 0 or n_y <= 0:
        raise ValueError("n_x and n_y must be positive.")
    if n_eq < 0 or n_ineq < 0:
        raise ValueError("n_eq and n_ineq must be nonnegative.")
    if n_eq > n_y:
        raise ValueError("n_eq cannot exceed n_y.")
    if n_samples <= 0 or n_points <= 0:
        raise ValueError("N_samples and N_points must be positive.")
    if n_points > n_samples:
        raise ValueError("N_points cannot exceed N_samples.")
    if x_l.shape != (n_x,) or x_u.shape != (n_x,):
        raise ValueError(f"x_L and x_U must both have shape ({n_x},).")


def _build_problem_generator(data_cfg: dict):
    try:
        from scripts.misc.nlp_generator import NLPGenerator
    except ImportError as exc:
        raise SystemExit(
            "The NLP DC3 runner needs the shared NLP generator dependencies, including `cvxpy` "
            "for convex label generation."
        ) from exc

    gen = NLPGenerator(
        n_y=int(data_cfg["n_y"]),
        n_x=int(data_cfg["n_x"]),
        n_eq=int(data_cfg["n_eq"]),
        n_ineq=int(data_cfg["n_ineq"]),
        seed=int(data_cfg["seed"]),
        is_diag_Q=bool(data_cfg.get("is_diag_Q", False)),
    )
    solver_name = resolve_solver_name(data_cfg, default="SCS")
    gen.set_solver(solver_name)
    gen.build_problem_data(
        x_L=np.asarray(data_cfg["x_L"], dtype=float),
        x_U=np.asarray(data_cfg["x_U"], dtype=float),
        q_diag_shift=float(data_cfg.get("q_diag_shift", 0.5)),
        nl_margin=float(data_cfg.get("nl_margin", 1.0)),
        bound_margin=float(data_cfg.get("bound_margin", 1.0)),
        bound_scale=float(data_cfg.get("bound_scale", 0.2)),
        param_scale=float(data_cfg.get("param_scale", 0.4)),
        preview_num_samples=int(data_cfg.get("N_points", data_cfg.get("N_samples", 0))),
    )
    if str(getattr(gen, "requested_solver", "cvxpy")).lower() == "cvxpy":
        gen.build_cvxpy_problem(solver=solver_name)
    return gen


def _generate_dataset(generator, data_cfg: dict):
    start_time = time.perf_counter()
    n_samples = int(data_cfg["N_samples"])
    n_points = int(data_cfg["N_points"])
    solver_name = resolve_solver_name(data_cfg, default="SCS")

    xs = generator.sample_parameters(n_samples)
    kept_x = []
    kept_y = []
    objectives = []
    status_counts = {}
    solve_time_sec = 0.0

    for x in xs:
        result = generator.solve_for_x(x)
        status = str(result["status"])
        status_counts[status] = status_counts.get(status, 0) + 1
        solve_time_sec += float(result.get("solve_time_sec") or 0.0)
        if status in ("optimal", "optimal_inaccurate") and result["y"] is not None:
            kept_x.append(np.asarray(x, dtype=np.float64))
            kept_y.append(np.asarray(result["y"], dtype=np.float64))
            objectives.append(float(result["objective"]) if result["objective"] is not None else np.nan)
            if len(kept_x) >= n_points:
                break

    if len(kept_x) < n_points:
        raise RuntimeError(
            f"Only collected {len(kept_x)} successful points out of requested {n_points}. "
            "Increase N_samples or switch solver settings."
        )

    X = np.asarray(kept_x, dtype=np.float64)
    Y = np.asarray(kept_y, dtype=np.float64)
    metadata = {
        "problem_type": "nlp",
        "n_x": int(data_cfg["n_x"]),
        "n_y": int(data_cfg["n_y"]),
        "n_eq": int(data_cfg["n_eq"]),
        "n_ineq": int(data_cfg["n_ineq"]),
        "N_samples": n_samples,
        "N_points": n_points,
        "solver": solver_name,
        "seed": int(data_cfg["seed"]),
        "objective_min": float(np.nanmin(objectives)) if objectives else np.nan,
        "objective_max": float(np.nanmax(objectives)) if objectives else np.nan,
        "objective_mean": float(np.nanmean(objectives)) if objectives else np.nan,
        "status_counts": status_counts,
        "optimizer_generation_wall_time_sec": solve_time_sec,
        "optimizer_generation_total_wall_time_sec": time.perf_counter() - start_time,
    }
    return X, Y, metadata, generator.get_problem_data()


def _load_dc3_method_module():
    if torch is None:
        raise RuntimeError("DC3 runner requires `torch`. Install PyTorch in your environment first.")

    module_name = "_nlpopt_dc3_method_runtime"
    if module_name in sys.modules:
        return sys.modules[module_name]

    dc3_dir = ROOT / "dc3"
    method_path = dc3_dir / "method.py"
    if not method_path.exists():
        raise FileNotFoundError(f"Could not find DC3 method file at {method_path}")

    if str(dc3_dir) not in sys.path:
        sys.path.insert(0, str(dc3_dir))

    if "utils" not in sys.modules:
        utils_stub = types.ModuleType("utils")
        utils_stub.my_hash = lambda string: hashlib.sha1(str(string).encode("utf-8")).hexdigest()
        utils_stub.str_to_bool = _str_to_bool
        sys.modules["utils"] = utils_stub

    if "setproctitle" not in sys.modules:
        setproctitle_stub = types.ModuleType("setproctitle")
        setproctitle_stub.setproctitle = lambda _title: None
        sys.modules["setproctitle"] = setproctitle_stub

    spec = importlib.util.spec_from_file_location(module_name, method_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load DC3 method module from {method_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _resolve_split_fracs(cfg_dict: dict) -> tuple[float, float, float]:
    train_frac = float(cfg_dict.get("train_frac", 0.8))
    if not 0.0 < train_frac < 1.0:
        raise ValueError("config.json must satisfy 0 < train_frac < 1 for the DC3 runner.")
    holdout = max(0.0, 1.0 - train_frac)
    requested_val = float(cfg_dict.get("val_frac", holdout / 2.0))
    valid_frac = min(max(requested_val, 0.0), holdout)
    test_frac = max(0.0, holdout - valid_frac)
    if holdout > 0.0 and test_frac == 0.0:
        valid_frac = holdout / 2.0
        test_frac = holdout - valid_frac
    if valid_frac <= 0.0 or test_frac <= 0.0:
        raise ValueError(
            "DC3 runner needs non-empty validation and test splits. "
            "Use train_frac < 1.0 so the remaining data can be split across both."
        )
    return train_frac, valid_frac, test_frac


def _build_dc3_args(cfg_dict: dict, *, use_completion: bool) -> dict:
    corr_train_steps = int(cfg_dict.get("dc3_corrTrainSteps", 10))
    corr_test_steps = int(cfg_dict.get("dc3_corrTestMaxSteps", corr_train_steps))
    corr_mode = str(cfg_dict.get("dc3_corrMode", "partial" if use_completion else "full")).lower()
    if corr_mode == "partial" and not use_completion:
        corr_mode = "full"

    return {
        "probType": "nlp",
        "epochs": int(cfg_dict.get("epochs", 1000)),
        "printEvery": max(1, int(cfg_dict.get("print_every", 10))),
        "batchSize": int(cfg_dict.get("batch_size", 200)),
        "lr": float(cfg_dict.get("learning_rate", 1e-4)),
        "hiddenSize": int(cfg_dict.get("hidden_size", 200)),
        "softWeight": float(cfg_dict.get("dc3_softWeight", cfg_dict.get("alpha_consistency", 10.0))),
        "softWeightEqFrac": float(cfg_dict.get("dc3_softWeightEqFrac", 0.5)),
        "useCompl": bool(use_completion),
        "useTrainCorr": bool(cfg_dict.get("dc3_useTrainCorr", True)),
        "useTestCorr": bool(cfg_dict.get("dc3_useTestCorr", True)),
        "corrMode": corr_mode,
        "corrTrainSteps": corr_train_steps,
        "corrTestMaxSteps": corr_test_steps,
        "corrEps": float(cfg_dict.get("dc3_corrEps", 1e-4)),
        "corrLr": float(cfg_dict.get("dc3_corrLr", 1e-11)),
        "corrMomentum": float(cfg_dict.get("dc3_corrMomentum", 0.5)),
        "corrGradClip": float(cfg_dict.get("dc3_corrGradClip", 1e10)),
        "trainGradClipNorm": float(cfg_dict.get("dc3_trainGradClipNorm", 1e3)),
        "saveAllStats": bool(cfg_dict.get("dc3_saveAllStats", True)),
        "resultsSaveFreq": int(cfg_dict.get("dc3_resultsSaveFreq", 50)),
        "seed": int(cfg_dict.get("seed", 42)),
    }


def _save_dir(dataset_dir: Path, data_cfg: dict, cfg_dict: dict, args: dict) -> Path:
    payload = {
        "framework": "dc3",
        "data": data_cfg,
        "config_seed": int(cfg_dict.get("seed", 42)),
        "args": args,
    }
    return dataset_dir / "dc3" / f"run_{_json_hash(payload)}_{time.strftime('%Y%m%d_%H%M%S')}"


def _summarize_stats(stats: dict) -> dict:
    summary = {}
    for key in (
        "train_loss",
        "valid_eval",
        "valid_ineq_max",
        "valid_eq_max",
        "valid_steps",
        "test_eval",
        "test_ineq_max",
        "test_eq_max",
        "test_steps",
    ):
        if key not in stats:
            continue
        arr = np.asarray(stats[key])
        if arr.ndim == 0:
            summary[key] = float(arr)
        elif arr.ndim == 1:
            summary[key] = float(np.mean(arr))
        else:
            summary[key] = float(np.mean(arr[-1]))
    first_value = np.asarray(next(iter(stats.values()))) if stats else None
    summary["epochs_recorded"] = int(first_value.shape[0]) if first_value is not None and first_value.ndim > 0 else int(bool(stats))
    return summary


def _mean_stat(epoch_stats: dict, key: str) -> float:
    values = np.asarray(epoch_stats.get(key, np.array([])))
    if values.size == 0:
        return float("nan")
    return float(np.mean(values))


def _max_stat(epoch_stats: dict, key: str) -> float:
    values = np.asarray(epoch_stats.get(key, np.array([])))
    if values.size == 0:
        return float("nan")
    return float(np.max(values))


def _dc3_consistency(epoch_stats: dict, prefix: str) -> float:
    return _mean_stat(epoch_stats, f"{prefix}_dist")


def _dc3_violation(epoch_stats: dict, prefix: str) -> float:
    values = []
    for suffix in ("ineq_max", "eq_max"):
        stat = _max_stat(epoch_stats, f"{prefix}_{suffix}")
        if np.isfinite(stat):
            values.append(stat)
    if not values:
        return float("nan")
    return float(max(values))


def _save_dc3_training_artifacts(stats: dict, solver_net, save_dir: str) -> None:
    with open(Path(save_dir) / "stats.dict", "wb") as fh:
        pickle.dump(stats, fh)
    with open(Path(save_dir) / "solver_net.dict", "wb") as fh:
        torch.save(solver_net.state_dict(), fh)


def _eval_net_with_timing(dc3_method, data, X, solver_net, args, prefix: str, stats: dict, phase_totals=None) -> None:
    eps_converge = args["corrEps"]
    make_prefix = lambda suffix: f"{prefix}_{suffix}"

    start_time = time.time()
    Y = solver_net(X)
    forward_end_time = time.time()

    Ycorr, steps = dc3_method.grad_steps_all(data, X, Y, args)
    corrected_end_time = time.time()

    Ynew = dc3_method.grad_steps(data, X, Y, args)
    raw_end_time = time.time()

    if phase_totals is not None:
        phase_totals["backbone"] += forward_end_time - start_time
        phase_totals["projection"] += raw_end_time - forward_end_time

    dc3_method.dict_agg(stats, make_prefix("time"), corrected_end_time - start_time, op="sum")
    dc3_method.dict_agg(stats, make_prefix("steps"), np.array([steps]))
    dc3_method.dict_agg(stats, make_prefix("loss"), dc3_method.total_loss(data, X, Ynew, args).detach().cpu().numpy())
    dc3_method.dict_agg(stats, make_prefix("eval"), data.obj_fn(Ycorr).detach().cpu().numpy())
    dc3_method.dict_agg(stats, make_prefix("dist"), torch.norm(Ycorr - Y, dim=1).detach().cpu().numpy())
    dc3_method.dict_agg(
        stats,
        make_prefix("ineq_max"),
        torch.max(data.ineq_dist(X, Ycorr), dim=1)[0].detach().cpu().numpy(),
    )
    dc3_method.dict_agg(
        stats,
        make_prefix("ineq_mean"),
        torch.mean(data.ineq_dist(X, Ycorr), dim=1).detach().cpu().numpy(),
    )
    dc3_method.dict_agg(
        stats,
        make_prefix("ineq_num_viol_0"),
        torch.sum(data.ineq_dist(X, Ycorr) > eps_converge, dim=1).detach().cpu().numpy(),
    )
    dc3_method.dict_agg(
        stats,
        make_prefix("ineq_num_viol_1"),
        torch.sum(data.ineq_dist(X, Ycorr) > 10 * eps_converge, dim=1).detach().cpu().numpy(),
    )
    dc3_method.dict_agg(
        stats,
        make_prefix("ineq_num_viol_2"),
        torch.sum(data.ineq_dist(X, Ycorr) > 100 * eps_converge, dim=1).detach().cpu().numpy(),
    )
    dc3_method.dict_agg(
        stats,
        make_prefix("eq_max"),
        torch.max(torch.abs(data.eq_resid(X, Ycorr)), dim=1)[0].detach().cpu().numpy(),
    )
    dc3_method.dict_agg(
        stats,
        make_prefix("eq_mean"),
        torch.mean(torch.abs(data.eq_resid(X, Ycorr)), dim=1).detach().cpu().numpy(),
    )
    dc3_method.dict_agg(
        stats,
        make_prefix("eq_num_viol_0"),
        torch.sum(torch.abs(data.eq_resid(X, Ycorr)) > eps_converge, dim=1).detach().cpu().numpy(),
    )
    dc3_method.dict_agg(
        stats,
        make_prefix("eq_num_viol_1"),
        torch.sum(torch.abs(data.eq_resid(X, Ycorr)) > 10 * eps_converge, dim=1).detach().cpu().numpy(),
    )
    dc3_method.dict_agg(
        stats,
        make_prefix("eq_num_viol_2"),
        torch.sum(torch.abs(data.eq_resid(X, Ycorr)) > 100 * eps_converge, dim=1).detach().cpu().numpy(),
    )
    dc3_method.dict_agg(
        stats,
        make_prefix("raw_time"),
        (raw_end_time - corrected_end_time) + (forward_end_time - start_time),
        op="sum",
    )
    dc3_method.dict_agg(stats, make_prefix("raw_eval"), data.obj_fn(Ynew).detach().cpu().numpy())
    dc3_method.dict_agg(
        stats,
        make_prefix("raw_ineq_max"),
        torch.max(data.ineq_dist(X, Ynew), dim=1)[0].detach().cpu().numpy(),
    )
    dc3_method.dict_agg(
        stats,
        make_prefix("raw_ineq_mean"),
        torch.mean(data.ineq_dist(X, Ynew), dim=1).detach().cpu().numpy(),
    )
    dc3_method.dict_agg(
        stats,
        make_prefix("raw_ineq_num_viol_0"),
        torch.sum(data.ineq_dist(X, Ynew) > eps_converge, dim=1).detach().cpu().numpy(),
    )
    dc3_method.dict_agg(
        stats,
        make_prefix("raw_ineq_num_viol_1"),
        torch.sum(data.ineq_dist(X, Ynew) > 10 * eps_converge, dim=1).detach().cpu().numpy(),
    )
    dc3_method.dict_agg(
        stats,
        make_prefix("raw_ineq_num_viol_2"),
        torch.sum(data.ineq_dist(X, Ynew) > 100 * eps_converge, dim=1).detach().cpu().numpy(),
    )
    dc3_method.dict_agg(
        stats,
        make_prefix("raw_eq_max"),
        torch.max(torch.abs(data.eq_resid(X, Ynew)), dim=1)[0].detach().cpu().numpy(),
    )
    dc3_method.dict_agg(
        stats,
        make_prefix("raw_eq_mean"),
        torch.mean(torch.abs(data.eq_resid(X, Ynew)), dim=1).detach().cpu().numpy(),
    )
    dc3_method.dict_agg(
        stats,
        make_prefix("raw_eq_num_viol_0"),
        torch.sum(torch.abs(data.eq_resid(X, Ynew)) > eps_converge, dim=1).detach().cpu().numpy(),
    )
    dc3_method.dict_agg(
        stats,
        make_prefix("raw_eq_num_viol_1"),
        torch.sum(torch.abs(data.eq_resid(X, Ynew)) > 10 * eps_converge, dim=1).detach().cpu().numpy(),
    )
    dc3_method.dict_agg(
        stats,
        make_prefix("raw_eq_num_viol_2"),
        torch.sum(torch.abs(data.eq_resid(X, Ynew)) > 100 * eps_converge, dim=1).detach().cpu().numpy(),
    )


def _predict_corrected(dc3_method, data, solver_net, args, X):
    Y = solver_net(X)
    Ycorr, _ = dc3_method.grad_steps_all(data, X, Y, args)
    return Ycorr


def _split_violation_components(data, X, Y) -> tuple[float, float, float]:
    eq_max = 0.0
    if data.neq > 0:
        eq_max = float(torch.max(torch.abs(data.eq_resid(X, Y))).item())

    lower, upper = data._bounds(X)
    bound_violation = torch.maximum(torch.clamp(lower - Y, min=0.0), torch.clamp(Y - upper, min=0.0))
    bound_max = float(torch.max(bound_violation).item()) if bound_violation.numel() > 0 else 0.0

    ineq_resid = data._nonlinear_resid(X, Y)
    ineq_violation = torch.clamp(ineq_resid, min=0.0)
    ineq_max = float(torch.max(ineq_violation).item()) if ineq_violation.numel() > 0 else 0.0
    return eq_max, ineq_max, bound_max


def _print_final_dc3_summary(dc3_method, data, solver_net, args: dict, profile: dict) -> dict:
    from torch.utils.data import DataLoader, TensorDataset

    batch_size = int(args["batchSize"])
    eval_sets = (
        TensorDataset(data.trainX, data.trainY),
        TensorDataset(data.validX, data.validY),
    )

    eq_max = 0.0
    ineq_max = 0.0
    bound_max = 0.0
    mse_total = 0.0
    mse_count = 0
    gap_total = 0.0
    gap_count = 0

    solver_net.eval()
    with torch.no_grad():
        for dataset in eval_sets:
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            for Xbatch, Ytrue in loader:
                Xbatch = Xbatch.to(dc3_method.DEVICE)
                Ytrue = Ytrue.to(dc3_method.DEVICE)
                Ypred = _predict_corrected(dc3_method, data, solver_net, args, Xbatch)

                batch_eq_max, batch_ineq_max, batch_bound_max = _split_violation_components(data, Xbatch, Ypred)
                eq_max = max(eq_max, batch_eq_max)
                ineq_max = max(ineq_max, batch_ineq_max)
                bound_max = max(bound_max, batch_bound_max)

                mse_total += float(torch.sum((Ypred - Ytrue) ** 2).item())
                mse_count += int(Ytrue.numel())

                pred_obj = data.obj_fn(Ypred)
                ref_obj = data.obj_fn(Ytrue)
                rel_gap = torch.abs(pred_obj - ref_obj) / torch.maximum(
                    torch.ones_like(ref_obj),
                    torch.abs(ref_obj),
                )
                gap_total += float(torch.sum(rel_gap).item())
                gap_count += int(rel_gap.numel())

    mse_final = mse_total / max(1, mse_count)
    rel_gap_final = gap_total / max(1, gap_count)

    timing_summary = summarize_timing_profile(profile)
    profiled_total = (
        float(timing_summary["backbone_total_sec"])
        + float(timing_summary["projection_total_sec"])
        + float(timing_summary["backward_total_sec"])
        + float(timing_summary["optimizer_total_sec"])
    )

    print("\n=== ORIGINAL constraint violation (max over train+val) ===")
    print(f"Equality   ||A y - (b+Bx)||_inf : {eq_max:.6e}")
    print(f"Inequality max(·,0)_inf         : {ineq_max:.6e}")
    print(f"Bounds     max(lb,ub)_inf       : {bound_max:.6e}\n")

    print("=== Supervised evaluation (against label solver) ===")
    print(f"MSE(y_tilde, y_true): {mse_final:.6e}\n")

    print("=== Optimality gap (relative objective difference vs label solver) ===")
    print(f"Relative objective gap: {rel_gap_final:.6e}\n")

    print("=== Profiled training time distribution ===")
    print(f"Training wall time: {float(profile['training_wall_time_sec']):.3f}s")
    print(
        f"Average epoch time ({timing_window_label(int(profile['epochs']))}): "
        f"train {float(timing_summary['avg_train_epoch_time_sec']):.3f}s "
        f"val {float(timing_summary['avg_val_epoch_time_sec']):.3f}s "
        f"total {float(timing_summary['avg_total_epoch_time_sec']):.3f}s"
    )
    print(
        f"Average batch time ({timing_window_label(int(profile['epochs']))}): "
        f"train {float(timing_summary['avg_train_batch_time_sec']):.4f}s "
        f"val {float(timing_summary['avg_val_batch_time_sec']):.4f}s "
        f"overall {float(timing_summary['avg_total_batch_time_sec']):.4f}s"
    )
    if profiled_total > 0.0:
        print(f"Backbone forward : {float(timing_summary['time_backbone_percent']):6.2f}% (train + val)")
        print(f"Projection       : {float(timing_summary['time_projection_percent']):6.2f}% (train + val)")
        print(f"Backward         : {float(timing_summary['time_backward_percent']):6.2f}% (train only)")
        print(f"Optimizer update : {float(timing_summary['time_optimizer_percent']):6.2f}% (train only)")
    print("")

    summary = {
        "eq_inf": eq_max,
        "ineq_inf": ineq_max,
        "bound_inf": bound_max,
        "mse_y_tilde_vs_label": mse_final,
        "relative_objective_gap": rel_gap_final,
        "training_wall_time_sec": float(profile["training_wall_time_sec"]),
    }
    summary.update(timing_summary)
    return summary


def _train_net_with_nlpopt_logging(dc3_method, data, args: dict, save_dir: str):
    from torch.utils.data import DataLoader, TensorDataset
    import torch.optim as optim

    solver_step = args["lr"]
    nepochs = int(args["epochs"])
    batch_size = int(args["batchSize"])

    train_dataset = TensorDataset(data.trainX)
    valid_dataset = TensorDataset(data.validX)
    test_dataset = TensorDataset(data.testX)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    train_eval_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    solver_net = dc3_method.NNSolver(data, args)
    solver_net.to(dc3_method.DEVICE)
    solver_opt = optim.Adam(solver_net.parameters(), lr=solver_step)

    stats = {}
    train_epoch_time_total = 0.0
    val_epoch_time_total = 0.0
    train_epoch_time_tracked = 0.0
    val_epoch_time_tracked = 0.0
    backbone_total_sec = 0.0
    projection_total_sec = 0.0
    backward_total_sec = 0.0
    optimizer_total_sec = 0.0
    backbone_tracked_total_sec = 0.0
    projection_tracked_total_sec = 0.0
    backward_tracked_total_sec = 0.0
    optimizer_tracked_total_sec = 0.0
    training_wall_t0 = time.time()
    for ep in range(nepochs):
        epoch_stats = {}
        val_phase_totals = {"backbone": 0.0, "projection": 0.0}

        solver_net.train()
        train_epoch_t0 = time.time()
        for Xtrain in train_loader:
            Xtrain = Xtrain[0].to(dc3_method.DEVICE)
            start_time = time.time()
            solver_opt.zero_grad()
            Yhat_train = solver_net(Xtrain)
            after_forward = time.time()
            Ynew_train = dc3_method.grad_steps(data, Xtrain, Yhat_train, args)
            after_projection = time.time()
            train_loss = dc3_method.total_loss(data, Xtrain, Ynew_train, args)
            after_loss = time.time()
            train_loss.sum().backward()
            after_backward = time.time()
            torch.nn.utils.clip_grad_norm_(
                solver_net.parameters(),
                max_norm=float(args.get("trainGradClipNorm", 1e3)),
            )
            solver_opt.step()
            after_optimizer = time.time()
            train_time = after_optimizer - start_time
            backbone_total_sec += after_forward - start_time
            projection_total_sec += after_loss - after_forward
            backward_total_sec += after_backward - after_loss
            optimizer_total_sec += after_optimizer - after_backward
            if should_track_epoch(ep, nepochs):
                backbone_tracked_total_sec += after_forward - start_time
                projection_tracked_total_sec += after_loss - after_forward
                backward_tracked_total_sec += after_backward - after_loss
                optimizer_tracked_total_sec += after_optimizer - after_backward
            dc3_method.dict_agg(epoch_stats, "train_loss", train_loss.detach().cpu().numpy())
            dc3_method.dict_agg(epoch_stats, "train_time", train_time, op="sum")
        train_epoch_elapsed = time.time() - train_epoch_t0
        train_epoch_time_total += train_epoch_elapsed
        if should_track_epoch(ep, nepochs):
            train_epoch_time_tracked += train_epoch_elapsed

        solver_net.eval()
        with torch.no_grad():
            val_epoch_t0 = time.time()
            for Xbatch in valid_loader:
                Xbatch = Xbatch[0].to(dc3_method.DEVICE)
                _eval_net_with_timing(dc3_method, data, Xbatch, solver_net, args, "valid", epoch_stats, val_phase_totals)
            val_epoch_elapsed = time.time() - val_epoch_t0
            val_epoch_time_total += val_epoch_elapsed
            backbone_total_sec += val_phase_totals["backbone"]
            projection_total_sec += val_phase_totals["projection"]
            if should_track_epoch(ep, nepochs):
                val_epoch_time_tracked += val_epoch_elapsed
                backbone_tracked_total_sec += val_phase_totals["backbone"]
                projection_tracked_total_sec += val_phase_totals["projection"]

            for Xbatch in train_eval_loader:
                Xbatch = Xbatch[0].to(dc3_method.DEVICE)
                _eval_net_with_timing(dc3_method, data, Xbatch, solver_net, args, "train", epoch_stats)
            for Xbatch in test_loader:
                Xbatch = Xbatch[0].to(dc3_method.DEVICE)
                _eval_net_with_timing(dc3_method, data, Xbatch, solver_net, args, "test", epoch_stats)

        if (ep % int(args.get("printEvery", 10))) == 0 or ep == nepochs - 1:
            print(
                f"ep {ep:05d} | "
                f"train loss {_mean_stat(epoch_stats, 'train_loss'):.6e} "
                f"obj {_mean_stat(epoch_stats, 'train_eval'):.6e} "
                f"cons {_dc3_consistency(epoch_stats, 'train'):.6e} "
                f"viol {_dc3_violation(epoch_stats, 'train'):.6e} || "
                f"val loss {_mean_stat(epoch_stats, 'valid_loss'):.6e} "
                f"obj {_mean_stat(epoch_stats, 'valid_eval'):.6e} "
                f"cons {_dc3_consistency(epoch_stats, 'valid'):.6e} "
                f"viol {_dc3_violation(epoch_stats, 'valid'):.6e}"
            )

        if args["saveAllStats"]:
            if ep == 0:
                for key, value in epoch_stats.items():
                    stats[key] = np.expand_dims(np.array(value), axis=0)
            else:
                for key, value in epoch_stats.items():
                    stats[key] = np.concatenate((stats[key], np.expand_dims(np.array(value), axis=0)))
        else:
            stats = epoch_stats

        if (ep % int(args["resultsSaveFreq"])) == 0:
            _save_dc3_training_artifacts(stats, solver_net, save_dir)

    training_wall_time_sec = time.time() - training_wall_t0
    _save_dc3_training_artifacts(stats, solver_net, save_dir)
    profile = {
        "training_wall_time_sec": training_wall_time_sec,
        "train_epoch_time_total_sec": train_epoch_time_total,
        "val_epoch_time_total_sec": val_epoch_time_total,
        "train_epoch_time_tracked_sec": train_epoch_time_tracked,
        "val_epoch_time_tracked_sec": val_epoch_time_tracked,
        "backbone_total_sec": backbone_total_sec,
        "projection_total_sec": projection_total_sec,
        "backward_total_sec": backward_total_sec,
        "optimizer_total_sec": optimizer_total_sec,
        "backbone_tracked_total_sec": backbone_tracked_total_sec,
        "projection_tracked_total_sec": projection_tracked_total_sec,
        "backward_tracked_total_sec": backward_tracked_total_sec,
        "optimizer_tracked_total_sec": optimizer_tracked_total_sec,
        "epochs": nepochs,
        "train_batches_per_epoch": len(train_loader),
        "val_batches_per_epoch": len(valid_loader),
    }
    return solver_net, stats, profile


def _choose_completion_indices(A: np.ndarray, *, seed: int) -> tuple[np.ndarray, np.ndarray]:
    n_eq, n_y = A.shape
    if n_eq <= 0 or n_y <= n_eq:
        return np.arange(n_y, dtype=np.int64), np.zeros((0,), dtype=np.int64)
    rng = np.random.default_rng(seed)
    best_other = None
    best_score = -np.inf
    for _ in range(1024):
        other = np.sort(rng.choice(n_y, size=n_eq, replace=False)).astype(np.int64)
        block = A[:, other]
        svals = np.linalg.svd(block, compute_uv=False)
        if svals.size == 0 or np.min(svals) <= 1e-8:
            continue
        score = float(np.min(svals) / np.max(svals))
        if score > best_score:
            best_score = score
            best_other = other
    if best_other is None:
        raise RuntimeError("Unable to find a stable invertible equality subset for DC3 completion.")
    partial = np.setdiff1d(np.arange(n_y, dtype=np.int64), best_other)
    return partial, best_other


class NLPDC3Problem:
    def __init__(
        self,
        problem_data: dict,
        X: np.ndarray,
        Y: np.ndarray,
        *,
        train_frac: float,
        valid_frac: float,
        test_frac: float,
        split_seed: int,
    ):
        if torch is None:
            raise RuntimeError("DC3 runner requires `torch`. Install PyTorch in your environment first.")

        rng = np.random.default_rng(split_seed)
        order = np.arange(X.shape[0])
        rng.shuffle(order)
        X = np.asarray(X[order], dtype=np.float64)
        Y = np.asarray(Y[order], dtype=np.float64)

        self._Q = torch.tensor(problem_data["Q"], dtype=torch.float64)
        self._c = torch.tensor(problem_data["c"], dtype=torch.float64)
        self._A = torch.tensor(problem_data["A"], dtype=torch.float64)
        self._b = torch.tensor(problem_data["b"], dtype=torch.float64)
        self._B = torch.tensor(problem_data["B"], dtype=torch.float64)
        self._a = torch.tensor(problem_data["a"], dtype=torch.float64)
        self._W = torch.tensor(problem_data["W"], dtype=torch.float64)
        self._W_sum = self._W + self._W.transpose(1, 2)
        self._beta = torch.tensor(problem_data["beta"], dtype=torch.float64)
        self._E = torch.tensor(problem_data["E"], dtype=torch.float64)
        self._l = torch.tensor(problem_data["l"], dtype=torch.float64)
        self._L = torch.tensor(problem_data["L"], dtype=torch.float64)
        self._u = torch.tensor(problem_data["u"], dtype=torch.float64)
        self._U = torch.tensor(problem_data["U"], dtype=torch.float64)
        self._X = torch.tensor(X, dtype=torch.float64)
        self._Y = torch.tensor(Y, dtype=torch.float64)

        self._xdim = int(X.shape[1])
        self._ydim = int(Y.shape[1])
        self._num = int(X.shape[0])
        self._neq = int(self._A.shape[0])
        self._nineq = int(self._a.shape[0] + 2 * self._ydim)
        self._nknowns = 0
        self._train_frac = float(train_frac)
        self._valid_frac = float(valid_frac)
        self._test_frac = float(test_frac)
        self._device = None
        self._corr_grad_clip = 1e10

        use_completion = self._neq > 0 and self._ydim > self._neq
        self._use_completion = use_completion
        if use_completion:
            partial, other = _choose_completion_indices(np.asarray(problem_data["A"], dtype=np.float64), seed=split_seed + 17)
            self._partial_vars = partial
            self._other_vars = other
            self._A_other_inv = torch.inverse(self._A[:, self._other_vars])
            self._A_partial = self._A[:, self._partial_vars]
            self._d_other_d_partial = -(self._A_other_inv @ self._A_partial)
            self._dy_dz = torch.zeros((self._ydim, self._partial_vars.shape[0]), dtype=torch.float64)
            self._dy_dz[self._partial_vars, :] = torch.eye(self._partial_vars.shape[0], dtype=torch.float64)
            self._dy_dz[self._other_vars, :] = self._d_other_d_partial
        else:
            self._partial_vars = np.arange(self._ydim, dtype=np.int64)
            self._other_vars = np.zeros((0,), dtype=np.int64)
            self._A_other_inv = torch.zeros((0, 0), dtype=torch.float64)
            self._A_partial = torch.zeros((self._neq, self._ydim), dtype=torch.float64)
            self._d_other_d_partial = torch.zeros((0, self._ydim), dtype=torch.float64)
            self._dy_dz = torch.eye(self._ydim, dtype=torch.float64)

    def configure_stabilization(self, *, corr_grad_clip: float) -> None:
        self._corr_grad_clip = float(corr_grad_clip)

    @staticmethod
    def _finite_or_zero(tensor: torch.Tensor) -> torch.Tensor:
        return torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)

    def _clip_tensor(self, tensor: torch.Tensor, clip_abs: float | None = None) -> torch.Tensor:
        tensor = self._finite_or_zero(tensor)
        if clip_abs is None or clip_abs <= 0.0:
            return tensor
        return torch.clamp(tensor, min=-float(clip_abs), max=float(clip_abs))

    def __str__(self):
        return f"CustomDC3-nlp-{self.ydim}-{self.nineq}-{self.neq}-{self.num}"

    def to(self, device):
        for name, value in list(self.__dict__.items()):
            if torch.is_tensor(value):
                setattr(self, name, value.to(device))
        self._device = device
        return self

    @property
    def xdim(self):
        return self._xdim

    @property
    def ydim(self):
        return self._ydim

    @property
    def num(self):
        return self._num

    @property
    def neq(self):
        return self._neq

    @property
    def nineq(self):
        return self._nineq

    @property
    def nknowns(self):
        return self._nknowns

    @property
    def partial_vars(self):
        return self._partial_vars

    @property
    def other_vars(self):
        return self._other_vars

    @property
    def partial_unknown_vars(self):
        return self._partial_vars

    @property
    def train_frac(self):
        return self._train_frac

    @property
    def valid_frac(self):
        return self._valid_frac

    @property
    def test_frac(self):
        return self._test_frac

    @property
    def trainX(self):
        return self._X[: int(self.num * self.train_frac)]

    @property
    def validX(self):
        start = int(self.num * self.train_frac)
        end = int(self.num * (self.train_frac + self.valid_frac))
        return self._X[start:end]

    @property
    def testX(self):
        start = int(self.num * (self.train_frac + self.valid_frac))
        return self._X[start:]

    @property
    def trainY(self):
        return self._Y[: int(self.num * self.train_frac)]

    @property
    def validY(self):
        start = int(self.num * self.train_frac)
        end = int(self.num * (self.train_frac + self.valid_frac))
        return self._Y[start:end]

    @property
    def testY(self):
        start = int(self.num * (self.train_frac + self.valid_frac))
        return self._Y[start:]

    @property
    def device(self):
        return self._device

    def _eq_rhs(self, X):
        if self._neq == 0:
            return torch.zeros((X.shape[0], 0), device=X.device, dtype=X.dtype)
        return self._b.unsqueeze(0) + X @ self._B.T

    def _ineq_rhs(self, X):
        return self._beta.unsqueeze(0) + X @ self._E.T

    def _bounds(self, X):
        lower = self._l.unsqueeze(0) + X @ self._L.T
        upper = self._u.unsqueeze(0) + X @ self._U.T
        return lower, upper

    def _squash_to_box(self, lower, upper, raw):
        span = torch.clamp(upper - lower, min=1e-6)
        return lower + 0.5 * span * (torch.tanh(raw) + 1.0)

    def obj_fn(self, Y):
        values = (0.5 * (Y @ self._Q) * Y + self._c * Y).sum(dim=1)
        return self._finite_or_zero(values)

    def eq_resid(self, X, Y):
        if self._neq == 0:
            return torch.zeros((X.shape[0], 0), device=Y.device, dtype=Y.dtype)
        return self._finite_or_zero(Y @ self._A.T - self._eq_rhs(X))

    def _nonlinear_resid(self, X, Y):
        exp_y = torch.exp(torch.clamp(Y, min=-20.0, max=12.0))
        exp_term = exp_y @ self._a.T
        quad_term = torch.einsum("bi,mij,bj->bm", Y, self._W, Y)
        return self._finite_or_zero(exp_term + quad_term - self._ineq_rhs(X))

    def ineq_resid(self, X, Y):
        lower, upper = self._bounds(X)
        lower_resid = lower - Y
        upper_resid = Y - upper
        return self._finite_or_zero(torch.cat([self._nonlinear_resid(X, Y), lower_resid, upper_resid], dim=1))

    def ineq_dist(self, X, Y):
        return self._finite_or_zero(torch.clamp(self.ineq_resid(X, Y), min=0.0))

    def eq_grad(self, X, Y):
        if self._neq == 0:
            return torch.zeros_like(Y)
        return self._clip_tensor(2.0 * (self.eq_resid(X, Y) @ self._A), self._corr_grad_clip)

    def ineq_grad(self, X, Y):
        nl_resid = self._nonlinear_resid(X, Y)
        nl_dist = torch.clamp(nl_resid, min=0.0)
        exp_y = torch.exp(torch.clamp(Y, min=-20.0, max=12.0))
        exp_grad = self._a.unsqueeze(0) * exp_y.unsqueeze(1)
        quad_grad = torch.einsum("bi,mij->bmj", Y, self._W_sum)
        grad = 2.0 * torch.sum(nl_dist.unsqueeze(-1) * (exp_grad + quad_grad), dim=1)

        lower, upper = self._bounds(X)
        lower_dist = torch.clamp(lower - Y, min=0.0)
        upper_dist = torch.clamp(Y - upper, min=0.0)
        grad = grad - 2.0 * lower_dist + 2.0 * upper_dist
        return self._clip_tensor(grad, self._corr_grad_clip)

    def ineq_partial_grad(self, X, Y):
        if not self._use_completion:
            return self.ineq_grad(X, Y)
        grad_full = self.ineq_grad(X, Y)
        grad_partial = grad_full @ self._dy_dz
        return self._clip_tensor(grad_partial @ self._dy_dz.T, self._corr_grad_clip)

    def process_output(self, X, Y):
        lower, upper = self._bounds(X)
        return self._finite_or_zero(self._squash_to_box(lower, upper, Y))

    def complete_partial(self, X, Z):
        if not self._use_completion:
            return self.process_output(X, Z)
        rhs = self._eq_rhs(X)
        lower, upper = self._bounds(X)
        z_scaled = self._squash_to_box(lower[:, self._partial_vars], upper[:, self._partial_vars], Z)
        Y = torch.zeros((X.shape[0], self.ydim), device=self.device, dtype=X.dtype)
        Y[:, self._partial_vars] = z_scaled
        Y[:, self._other_vars] = rhs @ self._A_other_inv.T + z_scaled @ self._d_other_d_partial.T
        return self._finite_or_zero(Y)


def run_case(case_dir: Path) -> int:
    if torch is None:
        raise SystemExit("DC3 runner requires `torch`. Install PyTorch before running this script.")

    data_cfg = _load_json(case_dir / "data.json")
    cfg_dict = _load_json(case_dir / "config.json")
    _validate_data_cfg(data_cfg)
    seed_torch_runtime(int(cfg_dict.get("seed", 42)))

    generator = _build_problem_generator(data_cfg)
    dataset = ensure_cached_dataset(
        case_dir,
        data_cfg,
        lambda: _generate_dataset(generator, data_cfg),
        force=bool(data_cfg.get("force_regenerate", False)),
    )
    problem_data = generator.get_problem_data()

    train_frac, valid_frac, test_frac = _resolve_split_fracs(cfg_dict)
    problem = NLPDC3Problem(
        problem_data,
        dataset.X,
        dataset.Y,
        train_frac=train_frac,
        valid_frac=valid_frac,
        test_frac=test_frac,
        split_seed=int(cfg_dict.get("seed", 42)),
    )

    dc3_method = _load_dc3_method_module()
    args = _build_dc3_args(cfg_dict, use_completion=problem._use_completion)
    save_dir = _save_dir(dataset.dataset_dir, data_cfg, cfg_dict, args)
    save_dir.mkdir(parents=True, exist_ok=True)

    problem.to(dc3_method.DEVICE)
    problem.configure_stabilization(corr_grad_clip=float(args.get("corrGradClip", 1e10)))

    with open(save_dir / "args.json", "w", encoding="utf-8") as fh:
        json.dump(args, fh, indent=2, sort_keys=True)
    with open(save_dir / "args.dict", "wb") as fh:
        pickle.dump(args, fh)
    with open(save_dir / "data.json", "w", encoding="utf-8") as fh:
        json.dump(data_cfg, fh, indent=2, sort_keys=True)
    with open(save_dir / "config.json", "w", encoding="utf-8") as fh:
        json.dump(cfg_dict, fh, indent=2, sort_keys=True)

    print(f"Dataset: {dataset.dataset_dir}")
    print(f"DC3 save dir: {save_dir}")
    print("Problem type: nlp")
    print(f"Device: {dc3_method.DEVICE}")
    print(
        f"Splits: train={problem.trainX.shape[0]}  valid={problem.validX.shape[0]}  "
        f"test={problem.testX.shape[0]}"
    )

    solver_net, stats, profile = _train_net_with_nlpopt_logging(dc3_method, problem, args, str(save_dir))
    final_metrics = _print_final_dc3_summary(dc3_method, problem, solver_net, args, profile)
    summary = _summarize_stats(stats)
    summary.update(final_metrics)
    summary["dataset_dir"] = str(dataset.dataset_dir)
    summary["save_dir"] = str(save_dir)
    summary["framework"] = "dc3"
    with open(save_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)

    print(f"[dc3] Saved: {save_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_case(Path(__file__).resolve().parent))
