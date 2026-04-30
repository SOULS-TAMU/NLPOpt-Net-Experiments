#!/usr/bin/env python3
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import pickle
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

import jax.numpy as jnp
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

from dc3 import default_args as dc3_default_args  # noqa: E402
from scripts.plot_utils.plotting import (  # noqa: E402
    save_objective_violation_plot,
    save_shadow_objective_violation_plot,
)
from scripts.testcase import poly_run as poly_nlpopt  # noqa: E402
from scripts.testcase import poly_run_dc3 as poly_torch  # noqa: E402
from scripts.factory.poly_factory import build_problem_generator as build_poly_problem_generator  # noqa: E402
from scripts.factory.poly_factory import build_problem_model as build_poly_problem_model  # noqa: E402
from scripts.factory.poly_factory import build_problem_model_from_data as build_poly_problem_model_from_data  # noqa: E402
from scripts.factory.poly_factory import normalize_problem_type as normalize_poly_problem_type  # noqa: E402
from scripts.factory.poly_factory import uses_nonconvex_generator as uses_nonconvex_poly_generator  # noqa: E402
from scripts.misc.optimizer_profile import history_optimizer_timing_fields  # noqa: E402
from scripts.misc.poly_dataset_cache import dataset_dir as poly_dataset_dir  # noqa: E402
from scripts.misc.poly_dataset_cache import ensure_cached_dataset as ensure_poly_dataset  # noqa: E402
from scripts.misc.json_io import load_json as _load_json_from_path, write_json_atomic  # noqa: E402
from scripts.misc.runtime_seed import seed_torch_runtime  # noqa: E402
from scripts.misc.training_timing import should_track_epoch, summarize_timing_profile, timing_window_label  # noqa: E402
from scripts.testcase import nlp_run as nlp_nlpopt  # noqa: E402
from scripts.testcase import nlp_run_dc3 as nlp_torch  # noqa: E402
from scripts.misc.nlp_dataset_cache import dataset_dir as nlp_dataset_dir  # noqa: E402
from scripts.misc.nlp_dataset_cache import ensure_cached_dataset as ensure_nlp_dataset  # noqa: E402

UNIFIED_CFG_KEYS = {"model", "run_multiple_seed", "num_seeds"}
MODEL_ALIASES = {
    "nlpopt": "nlpopt",
    "dc3": "dc3",
    "baseline": "baseline_nn",
    "baseline_nn": "baseline_nn",
    "nn": "baseline_nn",
    "eqnn": "baseline_eq_nn",
    "baseline_eq_nn": "baseline_eq_nn",
}


@dataclass(frozen=True)
class RunArtifacts:
    framework: str
    dataset_dir: Path
    run_dir: Path
    history_path: Path
    metrics_path: Path
    plot_path: Path


def _load_json(path: Path) -> dict:
    return _load_json_from_path(path)


def _write_json(path: Path, payload: dict) -> None:
    write_json_atomic(path, payload)


def _json_hash(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:10]


def _write_run_configs(run_dir: Path, data_cfg: dict, cfg_dict: dict, proj_cfg: dict) -> None:
    _write_json(run_dir / "data.json", data_cfg)
    _write_json(run_dir / "config.json", cfg_dict)
    _write_json(run_dir / "proj.json", proj_cfg)


def _append_family_metadata(
    dataset_dir: Path,
    *,
    mode: str,
    output_dir: Path,
    data_cfg: dict,
    cfg_dict: dict,
    proj_cfg: dict,
    framework: str | None = None,
    frameworks: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
    extra: dict | None = None,
) -> Path:
    family_dir = dataset_dir.parent
    metadata_path = family_dir / "metadata.json"
    payload = {"entries": []}
    if metadata_path.exists():
        payload = _load_json(metadata_path)
        if not isinstance(payload, dict) or not isinstance(payload.get("entries"), list):
            payload = {"entries": []}

    next_entry_id = 1 + max((int(entry.get("entry_id", 0)) for entry in payload["entries"]), default=0)
    entry = {
        "entry_id": next_entry_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "mode": mode,
        "dataset_dir": str(dataset_dir),
        "output_dir": str(output_dir),
        "problem_type": str(data_cfg["type"]).lower(),
        "framework": framework,
        "frameworks": list(frameworks) if frameworks is not None else None,
        "seeds": [int(seed) for seed in seeds] if seeds is not None else None,
        "data_config": copy.deepcopy(data_cfg),
        "config": copy.deepcopy(cfg_dict),
        "proj_config": copy.deepcopy(proj_cfg),
    }
    if extra:
        entry["extra"] = copy.deepcopy(extra)
    payload["entries"].append(entry)
    _write_json(metadata_path, payload)
    return metadata_path


def _str_to_bool(value):
    if isinstance(value, bool):
        return value
    lowered = str(value).lower()
    if lowered in {"false", "f", "0", "no", "n"}:
        return False
    if lowered in {"true", "t", "1", "yes", "y"}:
        return True
    raise ValueError(f"{value!r} is not a valid boolean value")


def _normalize_model_name(model_name: str) -> str:
    normalized = str(model_name).strip().lower()
    if normalized not in MODEL_ALIASES:
        raise ValueError(
            "Unsupported model value. Expected one of: "
            "nlpopt, dc3, baseline, baseline_nn, eqnn, baseline_eq_nn."
        )
    return MODEL_ALIASES[normalized]


def _framework_label(framework: str) -> str:
    return {
        "nlpopt": "NLPOpt",
        "dc3": "DC3",
        "baseline_nn": "Baseline NN",
        "baseline_eq_nn": "Baseline Eq. NN",
    }[framework]


def _framework_dir(dataset_dir: Path, framework: str) -> Path:
    return dataset_dir / framework


def _framework_multi_dir(dataset_dir: Path, framework: str) -> Path:
    return _framework_dir(dataset_dir, framework) / "multi"


def _framework_seed_dir(dataset_dir: Path, framework: str, seed: int) -> Path:
    return _framework_multi_dir(dataset_dir, framework) / str(int(seed))


def _comparison_dir(dataset_dir: Path) -> Path:
    return dataset_dir / "comparison"


def _directory_size_mb(path: Path) -> float:
    total_bytes = 0
    if path.exists():
        for child in path.iterdir():
            if child.is_file():
                total_bytes += int(child.stat().st_size)
    return float(total_bytes) / (1024.0 * 1024.0)


def _problem_family(data_cfg: dict) -> str:
    problem_type = str(data_cfg["type"]).strip().lower()
    if problem_type == "nlp":
        return "nlp"
    if problem_type in {"qp", "qcqp", "convex_qp_jaxmodel", "convex_qcqp_jaxmodel"}:
        return "poly"
    raise ValueError(f"Unsupported problem type '{data_cfg['type']}'.")


def _training_cfg_only(cfg_dict: dict) -> dict:
    return {key: value for key, value in cfg_dict.items() if key not in UNIFIED_CFG_KEYS}


def _data_value(data_cfg: dict, primary: str, *fallbacks: str):
    for key in (primary, *fallbacks):
        if key in data_cfg:
            return copy.deepcopy(data_cfg[key])
    return None


def _filtered_data_cfg(data_cfg: dict) -> dict:
    from scripts.misc.solver_config import resolve_solver_name

    family = _problem_family(data_cfg)
    n_x = _data_value(data_cfg, "n_x", "p")
    n_y = _data_value(data_cfg, "n_y", "n")
    n_eq = _data_value(data_cfg, "n_eq", "me")
    n_ineq = _data_value(data_cfg, "n_ineq", "mi")
    num_samples = _data_value(data_cfg, "num_samples", "N_points", "N_samples")
    common_missing = []
    if n_x is None:
        common_missing.append("n_x")
    if n_y is None:
        common_missing.append("n_y")
    if n_eq is None:
        common_missing.append("n_eq")
    if n_ineq is None:
        common_missing.append("n_ineq")
    if num_samples is None:
        common_missing.append("num_samples")
    for required_key in ("seed", "is_diag_Q", "x_L", "x_U"):
        if required_key not in data_cfg:
            common_missing.append(required_key)
    if "solver" not in data_cfg and "cvxpy_solver" not in data_cfg:
        common_missing.append("solver")
    if common_missing:
        raise ValueError(
            "data.json is missing required keys: "
            f"{', '.join(sorted(common_missing))}"
        )

    if family == "poly":
        filtered = {
            "type": str(data_cfg["type"]),
            "p": int(n_x),
            "n": int(n_y),
            "me": int(n_eq),
            "mi": int(n_ineq),
            "num_samples": int(num_samples),
            "seed": int(data_cfg["seed"]),
            "solver": resolve_solver_name(data_cfg, default="SCS"),
            "is_diag_Q": bool(data_cfg.get("is_diag_Q", False)),
            "bound_radius": float(data_cfg.get("bound_radius", 2.0)),
            "force_regenerate": bool(data_cfg.get("force_regenerate", False)),
            "x_L": [float(v) for v in data_cfg["x_L"]],
            "x_U": [float(v) for v in data_cfg["x_U"]],
        }
        return filtered

    return {
        "type": str(data_cfg["type"]),
        "n_x": int(n_x),
        "n_y": int(n_y),
        "n_eq": int(n_eq),
        "n_ineq": int(n_ineq),
        "N_samples": int(num_samples),
        "N_points": int(num_samples),
        "seed": int(data_cfg["seed"]),
        "solver": resolve_solver_name(data_cfg, default="SCS"),
        "is_diag_Q": bool(data_cfg.get("is_diag_Q", False)),
        "q_diag_shift": float(data_cfg.get("q_diag_shift", 0.5)),
        "nl_margin": float(data_cfg.get("nl_margin", 1.0)),
        "bound_margin": float(data_cfg.get("bound_margin", 1.0)),
        "bound_scale": float(data_cfg.get("bound_scale", 0.2)),
        "param_scale": float(data_cfg.get("param_scale", 0.4)),
        "force_regenerate": bool(data_cfg.get("force_regenerate", False)),
        "x_L": [float(v) for v in data_cfg["x_L"]],
        "x_U": [float(v) for v in data_cfg["x_U"]],
    }


def _case_workspace() -> Path:
    case_dir = ROOT / "test"
    case_dir.mkdir(parents=True, exist_ok=True)
    return case_dir


def _problem_shape_text(data_cfg: dict) -> str:
    problem_type = str(data_cfg["type"]).strip().upper()
    if _problem_family(data_cfg) == "nlp":
        return (
            f"{problem_type}  "
            f"n_x={int(data_cfg['n_x'])} n_y={int(data_cfg['n_y'])} "
            f"n_eq={int(data_cfg['n_eq'])} n_ineq={int(data_cfg['n_ineq'])}"
        )
    return (
        f"{problem_type}  "
        f"p={int(data_cfg['p'])} n={int(data_cfg['n'])} "
        f"me={int(data_cfg['me'])} mi={int(data_cfg['mi'])}"
    )


def _num_batches(num_items: int, batch_size: int) -> int:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    return max(1, (int(num_items) + int(batch_size) - 1) // int(batch_size))


def _print_run_header(case_dir: Path, data_cfg: dict, cfg_dict: dict, framework: str) -> None:
    data_cfg = _filtered_data_cfg(data_cfg)
    dataset_target = _resolve_dataset_dir(case_dir, data_cfg)
    print("")
    print("=" * 80)
    print(f"Unified runner | framework={_framework_label(framework)}")
    print(_problem_shape_text(data_cfg))
    print(f"Workspace: {case_dir}")
    print(f"Dataset target: {dataset_target}")
    print(
        "Config: "
        f"seed={int(cfg_dict.get('seed', 42))} "
        f"epochs={int(cfg_dict.get('epochs', 10))} "
        f"batch_size={int(cfg_dict.get('batch_size', 40))} "
        f"lr={float(cfg_dict.get('learning_rate', 1e-3)):.3e}"
    )
    print("=" * 80)


def _load_dc3_runtime_module(filename: str, module_name: str):
    if torch is None:
        raise RuntimeError("Torch is required for DC3, Baseline NN, and EqNN runs.")

    if module_name in sys.modules:
        return sys.modules[module_name]

    dc3_dir = ROOT / "dc3"
    module_path = dc3_dir / filename
    if not module_path.exists():
        raise FileNotFoundError(f"Could not find DC3 runtime file at {module_path}")

    if str(dc3_dir) not in sys.path:
        sys.path.insert(0, str(dc3_dir))

    utils_module = sys.modules.get("utils")
    if utils_module is None:
        utils_module = types.ModuleType("utils")
    if not hasattr(utils_module, "my_hash"):
        utils_module.my_hash = lambda string: hashlib.sha1(str(string).encode("utf-8")).hexdigest()
    if not hasattr(utils_module, "str_to_bool"):
        utils_module.str_to_bool = _str_to_bool
    if not hasattr(utils_module, "PFFunction"):
        class _UnavailablePFFunction:
            def __init__(self, *_args, **_kwargs):
                pass

            def __call__(self, *_args, **_kwargs):
                raise NotImplementedError("ACOPF power-flow completion is not supported in the unified runner.")

        utils_module.PFFunction = _UnavailablePFFunction
    sys.modules["utils"] = utils_module

    if "setproctitle" not in sys.modules:
        setproctitle_stub = types.ModuleType("setproctitle")
        setproctitle_stub.setproctitle = lambda _title: None
        sys.modules["setproctitle"] = setproctitle_stub

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load DC3 runtime module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _build_baseline_nn_args(cfg_dict: dict) -> dict:
    defaults = dc3_default_args.baseline_nn_default_args("simple")
    return {
        "probType": "simple",
        "epochs": int(cfg_dict.get("epochs", defaults["epochs"])),
        "printEvery": max(1, int(cfg_dict.get("print_every", 10))),
        "batchSize": int(cfg_dict.get("batch_size", defaults["batchSize"])),
        "lr": float(cfg_dict.get("learning_rate", defaults["lr"])),
        "hiddenSize": int(cfg_dict.get("hidden_size", defaults["hiddenSize"])),
        "softWeight": float(cfg_dict.get("baseline_softWeight", cfg_dict.get("dc3_softWeight", cfg_dict.get("alpha_consistency", defaults["softWeight"])))),
        "softWeightEqFrac": float(cfg_dict.get("dc3_softWeightEqFrac", defaults["softWeightEqFrac"])),
        "useTestCorr": bool(cfg_dict.get("dc3_useTestCorr", defaults["useTestCorr"])),
        "corrTestMaxSteps": int(cfg_dict.get("dc3_corrTestMaxSteps", defaults["corrTestMaxSteps"])),
        "corrEps": float(cfg_dict.get("dc3_corrEps", defaults["corrEps"])),
        "corrLr": float(cfg_dict.get("dc3_corrLr", defaults["corrLr"])),
        "corrMomentum": float(cfg_dict.get("dc3_corrMomentum", defaults["corrMomentum"])),
        "saveAllStats": bool(cfg_dict.get("dc3_saveAllStats", defaults["saveAllStats"])),
        "resultsSaveFreq": int(cfg_dict.get("dc3_resultsSaveFreq", defaults["resultsSaveFreq"])),
        "seed": int(cfg_dict.get("seed", 42)),
    }


def _build_baseline_eq_nn_args(cfg_dict: dict, *, use_completion: bool) -> dict:
    defaults = dc3_default_args.baseline_eq_nn_default_args("simple")
    corr_mode = str(cfg_dict.get("dc3_corrMode", defaults["corrMode"])).lower()
    if corr_mode == "partial" and not use_completion:
        corr_mode = "full"
    return {
        "probType": "simple",
        "epochs": int(cfg_dict.get("epochs", defaults["epochs"])),
        "printEvery": max(1, int(cfg_dict.get("print_every", 10))),
        "batchSize": int(cfg_dict.get("batch_size", defaults["batchSize"])),
        "lr": float(cfg_dict.get("learning_rate", defaults["lr"])),
        "hiddenSize": int(cfg_dict.get("hidden_size", defaults["hiddenSize"])),
        "softWeightEqFrac": float(cfg_dict.get("dc3_softWeightEqFrac", defaults["softWeightEqFrac"])),
        "useTestCorr": bool(cfg_dict.get("dc3_useTestCorr", defaults["useTestCorr"])),
        "corrMode": corr_mode,
        "corrTestMaxSteps": int(cfg_dict.get("dc3_corrTestMaxSteps", defaults["corrTestMaxSteps"])),
        "corrEps": float(cfg_dict.get("dc3_corrEps", defaults["corrEps"])),
        "corrLr": float(cfg_dict.get("dc3_corrLr", defaults["corrLr"])),
        "corrMomentum": float(cfg_dict.get("dc3_corrMomentum", defaults["corrMomentum"])),
        "saveAllStats": bool(cfg_dict.get("dc3_saveAllStats", defaults["saveAllStats"])),
        "resultsSaveFreq": int(cfg_dict.get("dc3_resultsSaveFreq", defaults["resultsSaveFreq"])),
        "seed": int(cfg_dict.get("seed", 42)),
    }


def _framework_save_dir(dataset_dir: Path, framework: str, data_cfg: dict, cfg_dict: dict, args: dict) -> Path:
    payload = {
        "framework": framework,
        "data": data_cfg,
        "config_seed": int(cfg_dict.get("seed", 42)),
        "args": args,
    }
    return dataset_dir / framework / f"run_{_json_hash(payload)}_{time.strftime('%Y%m%d_%H%M%S')}"


def _save_model_artifacts(stats: dict, solver_net, save_dir: Path) -> None:
    with open(save_dir / "stats.dict", "wb") as fh:
        pickle.dump(stats, fh)
    with open(save_dir / "solver_net.dict", "wb") as fh:
        torch.save(solver_net.state_dict(), fh)


def _dict_agg(stats: dict, key: str, value, *, op: str = "concat") -> None:
    if key in stats:
        if op == "sum":
            stats[key] += value
        elif op == "concat":
            stats[key] = np.concatenate((stats[key], value), axis=0)
        else:
            raise NotImplementedError(op)
    else:
        stats[key] = value


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


def _torch_consistency(epoch_stats: dict, prefix: str) -> float:
    return _mean_stat(epoch_stats, f"{prefix}_dist")


def _torch_violation(epoch_stats: dict, prefix: str) -> float:
    values = []
    for suffix in ("ineq_max", "eq_max"):
        value = _max_stat(epoch_stats, f"{prefix}_{suffix}")
        if np.isfinite(value):
            values.append(value)
    if not values:
        return float("nan")
    return float(max(values))


def _save_epoch_summary(ep: int, epoch_stats: dict) -> None:
    print(
        f"ep {ep:05d} | "
        f"train loss {_mean_stat(epoch_stats, 'train_loss'):.6e} "
        f"obj {_mean_stat(epoch_stats, 'train_eval'):.6e} "
        f"cons {_torch_consistency(epoch_stats, 'train'):.6e} "
        f"viol {_torch_violation(epoch_stats, 'train'):.6e} || "
        f"val loss {_mean_stat(epoch_stats, 'valid_loss'):.6e} "
        f"obj {_mean_stat(epoch_stats, 'valid_eval'):.6e} "
        f"cons {_torch_consistency(epoch_stats, 'valid'):.6e} "
        f"viol {_torch_violation(epoch_stats, 'valid'):.6e}"
    )


def _epoch_mean_series(stats: dict, key: str) -> list[float]:
    values = np.asarray(stats[key])
    if values.ndim == 0:
        return [float(values)]
    if values.ndim == 1:
        return [float(np.mean(values))]
    return [float(np.mean(values[idx])) for idx in range(values.shape[0])]


def _epoch_max_series(stats: dict, key: str) -> list[float]:
    values = np.asarray(stats[key])
    if values.ndim == 0:
        return [float(values)]
    if values.ndim == 1:
        return [float(np.max(values))]
    return [float(np.max(values[idx])) for idx in range(values.shape[0])]


def _epoch_worst_gap_pct_series(stats: dict, key: str, ref_values) -> list[float]:
    pred = np.asarray(stats[key], dtype=float)
    ref = np.asarray(ref_values, dtype=float).reshape(-1)
    if pred.ndim == 0:
        return [float("nan")]
    if pred.ndim == 1:
        pred = np.expand_dims(pred, axis=0)
    if pred.shape[1] != ref.shape[0]:
        raise ValueError(f"Stats key '{key}' has width {pred.shape[1]}, expected {ref.shape[0]} labels.")
    denom = np.maximum(1.0, np.abs(ref))[None, :]
    gaps = 100.0 * np.abs(pred - ref[None, :]) / denom
    return [float(np.nanmax(gaps[idx])) for idx in range(gaps.shape[0])]


def _final_epoch_values(stats: dict, key: str) -> np.ndarray:
    if key not in stats:
        return np.asarray([], dtype=float)
    arr = np.asarray(stats[key], dtype=float)
    if arr.ndim == 0:
        return np.asarray([float(arr)], dtype=float)
    if arr.ndim == 1:
        return arr.astype(float)
    return np.asarray(arr[-1], dtype=float)


def _weighted_final_mean(
    stats: dict,
    train_key: str,
    val_key: str,
    *,
    train_count: int,
    val_count: int,
) -> float:
    weighted_total = 0.0
    total_weight = 0
    for key, count in ((train_key, int(train_count)), (val_key, int(val_count))):
        values = _final_epoch_values(stats, key)
        if values.size == 0:
            continue
        weighted_total += float(np.mean(values)) * count
        total_weight += count
    return float(weighted_total / total_weight) if total_weight > 0 else float("nan")


def _combined_final_max(stats: dict, train_key: str, val_key: str) -> float:
    maxima = []
    for key in (train_key, val_key):
        values = _final_epoch_values(stats, key)
        if values.size > 0:
            maxima.append(float(np.max(values)))
    return float(max(maxima)) if maxima else float("nan")


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


def _reference_objectives(problem) -> tuple[float, float]:
    train_ref = float(torch.mean(problem.obj_fn(problem.trainY)).item())
    val_ref = float(torch.mean(problem.obj_fn(problem.validY)).item())
    return train_ref, val_ref


def _save_history_and_plot(
    save_dir: Path,
    framework: str,
    stats: dict,
    problem,
    *,
    extra_history: dict | None = None,
) -> Path:
    if "train_eval" not in stats or "valid_eval" not in stats:
        raise ValueError("Missing train/valid objective histories in stats.")

    train_ref, val_ref = _reference_objectives(problem)
    train_ref_obj_values = problem.obj_fn(problem.trainY).detach().cpu().numpy()
    val_ref_obj_values = problem.obj_fn(problem.validY).detach().cpu().numpy()
    train_viol = np.maximum(_epoch_max_series(stats, "train_ineq_max"), _epoch_max_series(stats, "train_eq_max"))
    val_viol = np.maximum(_epoch_max_series(stats, "valid_ineq_max"), _epoch_max_series(stats, "valid_eq_max"))
    train_gap_pct = _epoch_worst_gap_pct_series(stats, "train_eval", train_ref_obj_values)
    val_gap_pct = _epoch_worst_gap_pct_series(stats, "valid_eval", val_ref_obj_values)
    history = {
        "epochs": list(range(len(_epoch_mean_series(stats, "train_eval")))),
        "train_objective": _epoch_mean_series(stats, "train_eval"),
        "val_objective": _epoch_mean_series(stats, "valid_eval"),
        "train_worst_relative_gap_pct": train_gap_pct,
        "val_worst_relative_gap_pct": val_gap_pct,
        "train_violation": [float(v) for v in train_viol],
        "val_violation": [float(v) for v in val_viol],
        "train_reference_objective": train_ref,
        "val_reference_objective": val_ref,
        "framework": framework,
    }
    if extra_history:
        history.update(extra_history)

    history_path = save_dir / "run_history.json"
    _write_json(history_path, history)
    plot_path = save_dir / "compare_metrics.png"
    try:
        save_objective_violation_plot(
            plot_path,
            epochs=history["epochs"],
            train_gap_pct=history["train_worst_relative_gap_pct"],
            val_gap_pct=history["val_worst_relative_gap_pct"],
            train_violation=history["train_violation"],
            val_violation=history["val_violation"],
            title=_framework_label(framework),
            series_label=_framework_label(framework),
        )
    except Exception as exc:
        warning_path = save_dir / "compare_metrics.error.txt"
        warning_path.write_text(f"Plot generation failed: {exc}\n", encoding="utf-8")
        print(f"[plot] Skipped for {framework}: {exc}")
    return history_path


def _iter_xy_batches(X, Y, batch_size: int):
    total = int(X.shape[0])
    for start in range(0, total, batch_size):
        stop = min(start + batch_size, total)
        yield X[start:stop], Y[start:stop]


def _violation_components(problem, X, Y):
    if int(problem.neq) > 0:
        eq_abs = torch.abs(problem.eq_resid(X, Y))
    else:
        eq_abs = torch.zeros((Y.shape[0], 0), dtype=Y.dtype, device=Y.device)
    lower, upper = problem._bounds(X)
    lower_violation = torch.clamp(lower - Y, min=0.0)
    upper_violation = torch.clamp(Y - upper, min=0.0)
    bound_violation = torch.maximum(lower_violation, upper_violation)

    if hasattr(problem, "_problem_type"):
        if str(problem._problem_type) == "qp":
            ineq_resid = problem._affine_ineq_resid(X, Y)
        else:
            ineq_resid = problem._quad_ineq_resid(X, Y)
    else:
        ineq_resid = problem._nonlinear_resid(X, Y)
    ineq_violation = torch.clamp(ineq_resid, min=0.0)
    return eq_abs, ineq_violation, bound_violation


def _split_violation_components(problem, X, Y) -> tuple[float, float, float]:
    eq_abs, ineq_violation, bound_violation = _violation_components(problem, X, Y)
    eq_max = float(torch.max(eq_abs).item()) if eq_abs.numel() > 0 else 0.0
    bound_max = float(torch.max(bound_violation).item()) if bound_violation.numel() > 0 else 0.0
    ineq_max = float(torch.max(ineq_violation).item()) if ineq_violation.numel() > 0 else 0.0
    return eq_max, ineq_max, bound_max


def _profile_batch_size(profile: dict, problem) -> int:
    if "batch_size" in profile:
        return max(1, int(profile["batch_size"]))
    if "batchSize" in profile:
        return max(1, int(profile["batchSize"]))
    train_batches = max(1, int(profile.get("train_batches_per_epoch", 1)))
    return max(1, int(np.ceil(int(problem.trainX.shape[0]) / train_batches)))


def _print_final_supervised_summary(
    problem,
    predict_corrected_fn: Callable,
    profile: dict,
    *,
    predict_raw_fn: Callable | None = None,
) -> dict:
    batch_size = _profile_batch_size(profile, problem)
    eq_max = 0.0
    ineq_max = 0.0
    bound_max = 0.0
    eq_sum = 0.0
    eq_count = 0
    ineq_sum = 0.0
    ineq_count = 0
    mse_total = 0.0
    mse_count = 0
    gap_total = 0.0
    gap_count = 0
    objective_total = 0.0
    objective_count = 0
    consistency_total = 0.0
    consistency_count = 0

    with torch.no_grad():
        for X, Y in ((problem.trainX, problem.trainY), (problem.validX, problem.validY)):
            for Xbatch, Ytrue in _iter_xy_batches(X, Y, batch_size):
                Ypred = predict_corrected_fn(Xbatch)
                eq_abs, ineq_violation, bound_violation = _violation_components(problem, Xbatch, Ypred)
                batch_eq, batch_ineq, batch_bound = _split_violation_components(problem, Xbatch, Ypred)
                eq_max = max(eq_max, batch_eq)
                ineq_max = max(ineq_max, batch_ineq)
                bound_max = max(bound_max, batch_bound)
                if eq_abs.numel() > 0:
                    eq_sum += float(torch.sum(eq_abs).item())
                    eq_count += int(eq_abs.numel())
                if ineq_violation.numel() > 0:
                    ineq_sum += float(torch.sum(ineq_violation).item())
                    ineq_count += int(ineq_violation.numel())

                mse_total += float(torch.sum((Ypred - Ytrue) ** 2).item())
                mse_count += int(Ytrue.numel())

                pred_obj = problem.obj_fn(Ypred)
                ref_obj = problem.obj_fn(Ytrue)
                objective_total += float(torch.sum(pred_obj).item())
                objective_count += int(pred_obj.numel())
                rel_gap = torch.abs(pred_obj - ref_obj) / torch.maximum(torch.ones_like(ref_obj), torch.abs(ref_obj))
                gap_total += float(torch.sum(rel_gap).item())
                gap_count += int(rel_gap.numel())
                if predict_raw_fn is not None:
                    Yraw = predict_raw_fn(Xbatch)
                    consistency_total += float(torch.sum(torch.norm(Ypred - Yraw, dim=1)).item())
                    consistency_count += int(Ypred.shape[0])

    mse_final = mse_total / max(1, mse_count)
    rel_gap_final = gap_total / max(1, gap_count)
    objective_value = objective_total / max(1, objective_count)
    mean_eq = eq_sum / max(1, eq_count)
    mean_ineq = ineq_sum / max(1, ineq_count)
    consistency_value = consistency_total / max(1, consistency_count) if consistency_count > 0 else float("nan")

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
        "objective_value": objective_value,
        "max_equality": eq_max,
        "mean_equality": mean_eq,
        "max_inequality": ineq_max,
        "mean_inequality": mean_ineq,
        "consistency": consistency_value,
        "eq_inf": eq_max,
        "ineq_inf": ineq_max,
        "bound_inf": bound_max,
        "mse_y_tilde_vs_label": mse_final,
        "relative_objective_gap": rel_gap_final,
        "training_wall_time_sec": float(profile["training_wall_time_sec"]),
    }
    summary.update(timing_summary)
    return summary


def _eval_baseline_nn_with_timing(module, data, X, solver_net, args, prefix: str, stats: dict, phase_totals=None) -> None:
    eps_converge = args["corrEps"]
    make_prefix = lambda suffix: f"{prefix}_{suffix}"

    start_time = time.time()
    Y = solver_net(X)
    forward_end_time = time.time()
    Ycorr, steps = module.grad_steps_all(data, X, Y, args)
    corrected_end_time = time.time()

    if phase_totals is not None:
        phase_totals["backbone"] += forward_end_time - start_time
        phase_totals["projection"] += corrected_end_time - forward_end_time

    _dict_agg(stats, make_prefix("time"), corrected_end_time - start_time, op="sum")
    _dict_agg(stats, make_prefix("steps"), np.array([steps]))
    _dict_agg(stats, make_prefix("loss"), module.softloss(data, X, Y, args).detach().cpu().numpy())
    _dict_agg(stats, make_prefix("eval"), data.obj_fn(Ycorr).detach().cpu().numpy())
    _dict_agg(stats, make_prefix("dist"), torch.norm(Ycorr - Y, dim=1).detach().cpu().numpy())
    _dict_agg(stats, make_prefix("ineq_max"), torch.max(data.ineq_dist(X, Ycorr), dim=1)[0].detach().cpu().numpy())
    _dict_agg(stats, make_prefix("ineq_mean"), torch.mean(data.ineq_dist(X, Ycorr), dim=1).detach().cpu().numpy())
    _dict_agg(stats, make_prefix("ineq_num_viol_0"), torch.sum(data.ineq_dist(X, Ycorr) > eps_converge, dim=1).detach().cpu().numpy())
    _dict_agg(stats, make_prefix("eq_max"), torch.max(torch.abs(data.eq_resid(X, Ycorr)), dim=1)[0].detach().cpu().numpy())
    _dict_agg(stats, make_prefix("eq_mean"), torch.mean(torch.abs(data.eq_resid(X, Ycorr)), dim=1).detach().cpu().numpy())
    _dict_agg(stats, make_prefix("eq_num_viol_0"), torch.sum(torch.abs(data.eq_resid(X, Ycorr)) > eps_converge, dim=1).detach().cpu().numpy())
    _dict_agg(stats, make_prefix("raw_time"), forward_end_time - start_time, op="sum")
    _dict_agg(stats, make_prefix("raw_eval"), data.obj_fn(Y).detach().cpu().numpy())
    _dict_agg(stats, make_prefix("raw_ineq_max"), torch.max(data.ineq_dist(X, Y), dim=1)[0].detach().cpu().numpy())
    _dict_agg(stats, make_prefix("raw_eq_max"), torch.max(torch.abs(data.eq_resid(X, Y)), dim=1)[0].detach().cpu().numpy())


def _train_baseline_nn_with_logging(module, data, args: dict, save_dir: Path):
    from torch.utils.data import DataLoader, TensorDataset
    import torch.optim as optim

    batch_size = int(args["batchSize"])
    nepochs = int(args["epochs"])
    print_every = max(1, int(args.get("printEvery", 10)))
    train_loader = DataLoader(TensorDataset(data.trainX), batch_size=batch_size, shuffle=True)
    train_eval_loader = DataLoader(TensorDataset(data.trainX), batch_size=batch_size, shuffle=False)
    valid_loader = DataLoader(TensorDataset(data.validX), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(data.testX), batch_size=batch_size, shuffle=False)

    solver_net = module.NNSolver(data, args)
    solver_net.to(module.DEVICE)
    solver_opt = optim.Adam(solver_net.parameters(), lr=float(args["lr"]))

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
        solver_net.train()
        train_epoch_t0 = time.time()
        for (Xtrain,) in train_loader:
            Xtrain = Xtrain.to(module.DEVICE)
            start_time = time.time()
            solver_opt.zero_grad()
            Yhat = solver_net(Xtrain)
            after_forward = time.time()
            train_loss = module.softloss(data, Xtrain, Yhat, args)
            after_loss = time.time()
            train_loss.sum().backward()
            after_backward = time.time()
            solver_opt.step()
            after_optimizer = time.time()

            backbone_total_sec += after_forward - start_time
            backward_total_sec += after_backward - after_loss
            optimizer_total_sec += after_optimizer - after_backward
            if should_track_epoch(ep, nepochs):
                backbone_tracked_total_sec += after_forward - start_time
                backward_tracked_total_sec += after_backward - after_loss
                optimizer_tracked_total_sec += after_optimizer - after_backward
            _dict_agg(epoch_stats, "train_loss", train_loss.detach().cpu().numpy())
            _dict_agg(epoch_stats, "train_time", after_optimizer - start_time, op="sum")
        train_epoch_elapsed = time.time() - train_epoch_t0
        train_epoch_time_total += train_epoch_elapsed
        if should_track_epoch(ep, nepochs):
            train_epoch_time_tracked += train_epoch_elapsed

        solver_net.eval()
        with torch.no_grad():
            val_phase_totals = {"backbone": 0.0, "projection": 0.0}
            val_epoch_t0 = time.time()
            for (Xvalid,) in valid_loader:
                Xvalid = Xvalid.to(module.DEVICE)
                _eval_baseline_nn_with_timing(module, data, Xvalid, solver_net, args, "valid", epoch_stats, val_phase_totals)
            val_epoch_elapsed = time.time() - val_epoch_t0
            val_epoch_time_total += val_epoch_elapsed
            backbone_total_sec += val_phase_totals["backbone"]
            projection_total_sec += val_phase_totals["projection"]
            if should_track_epoch(ep, nepochs):
                val_epoch_time_tracked += val_epoch_elapsed
                backbone_tracked_total_sec += val_phase_totals["backbone"]
                projection_tracked_total_sec += val_phase_totals["projection"]

            for (Xtrain_eval,) in train_eval_loader:
                Xtrain_eval = Xtrain_eval.to(module.DEVICE)
                _eval_baseline_nn_with_timing(module, data, Xtrain_eval, solver_net, args, "train", epoch_stats)
            for (Xtest,) in test_loader:
                Xtest = Xtest.to(module.DEVICE)
                _eval_baseline_nn_with_timing(module, data, Xtest, solver_net, args, "test", epoch_stats)

        if (ep % print_every) == 0 or ep == nepochs - 1:
            _save_epoch_summary(ep, epoch_stats)

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
            _save_model_artifacts(stats, solver_net, save_dir)

    _save_model_artifacts(stats, solver_net, save_dir)
    return solver_net, stats, {
        "training_wall_time_sec": time.time() - training_wall_t0,
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
        "batch_size": batch_size,
    }


def _eval_baseline_eq_nn_with_timing(module, data, X, Ytarget, solver_net, args, prefix: str, stats: dict, phase_totals=None) -> None:
    import torch.nn as nn

    eps_converge = args["corrEps"]
    make_prefix = lambda suffix: f"{prefix}_{suffix}"

    start_time = time.time()
    Z = solver_net(X)
    forward_end_time = time.time()
    Y = module.complete_f(data, X, Z, args)
    complete_end_time = time.time()
    Ycorr, steps = module.grad_steps_all(data, X, Y, args)
    corrected_end_time = time.time()

    if phase_totals is not None:
        phase_totals["backbone"] += forward_end_time - start_time
        phase_totals["projection"] += corrected_end_time - forward_end_time

    loss = nn.MSELoss(reduction="none")(Z, Ytarget).sum(dim=1)
    _dict_agg(stats, make_prefix("time"), corrected_end_time - start_time, op="sum")
    _dict_agg(stats, make_prefix("steps"), np.array([steps]))
    _dict_agg(stats, make_prefix("loss"), loss.detach().cpu().numpy())
    _dict_agg(stats, make_prefix("eval"), data.obj_fn(Ycorr).detach().cpu().numpy())
    _dict_agg(stats, make_prefix("dist"), torch.norm(Ycorr - Y, dim=1).detach().cpu().numpy())
    _dict_agg(stats, make_prefix("ineq_max"), torch.max(data.ineq_dist(X, Ycorr), dim=1)[0].detach().cpu().numpy())
    _dict_agg(stats, make_prefix("ineq_mean"), torch.mean(data.ineq_dist(X, Ycorr), dim=1).detach().cpu().numpy())
    _dict_agg(stats, make_prefix("ineq_num_viol_0"), torch.sum(data.ineq_dist(X, Ycorr) > eps_converge, dim=1).detach().cpu().numpy())
    _dict_agg(stats, make_prefix("eq_max"), torch.max(torch.abs(data.eq_resid(X, Ycorr)), dim=1)[0].detach().cpu().numpy())
    _dict_agg(stats, make_prefix("eq_mean"), torch.mean(torch.abs(data.eq_resid(X, Ycorr)), dim=1).detach().cpu().numpy())
    _dict_agg(stats, make_prefix("eq_num_viol_0"), torch.sum(torch.abs(data.eq_resid(X, Ycorr)) > eps_converge, dim=1).detach().cpu().numpy())
    _dict_agg(stats, make_prefix("raw_time"), complete_end_time - start_time, op="sum")
    _dict_agg(stats, make_prefix("raw_eval"), data.obj_fn(Y).detach().cpu().numpy())
    _dict_agg(stats, make_prefix("raw_ineq_max"), torch.max(data.ineq_dist(X, Y), dim=1)[0].detach().cpu().numpy())
    _dict_agg(stats, make_prefix("raw_eq_max"), torch.max(torch.abs(data.eq_resid(X, Y)), dim=1)[0].detach().cpu().numpy())


def _train_baseline_eq_nn_with_logging(module, data, args: dict, save_dir: Path):
    from torch.utils.data import DataLoader, TensorDataset
    import torch.nn as nn
    import torch.optim as optim

    batch_size = int(args["batchSize"])
    nepochs = int(args["epochs"])
    print_every = max(1, int(args.get("printEvery", 10)))
    train_dataset = TensorDataset(data.trainX, data.trainY[:, data.partial_unknown_vars])
    valid_dataset = TensorDataset(data.validX, data.validY[:, data.partial_unknown_vars])
    test_dataset = TensorDataset(data.testX, data.testY[:, data.partial_unknown_vars])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    train_eval_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    solver_net = module.NNSolver(data, args)
    solver_net.to(module.DEVICE)
    solver_opt = optim.Adam(solver_net.parameters(), lr=float(args["lr"]))

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
        solver_net.train()
        train_epoch_t0 = time.time()
        for Xtrain, Ytarget in train_loader:
            Xtrain = Xtrain.to(module.DEVICE)
            Ytarget = Ytarget.to(module.DEVICE)
            start_time = time.time()
            solver_opt.zero_grad()
            Z = solver_net(Xtrain)
            after_forward = time.time()
            train_loss = nn.MSELoss(reduction="none")(Z, Ytarget).sum(dim=1)
            after_loss = time.time()
            train_loss.sum().backward()
            after_backward = time.time()
            solver_opt.step()
            after_optimizer = time.time()

            backbone_total_sec += after_forward - start_time
            backward_total_sec += after_backward - after_loss
            optimizer_total_sec += after_optimizer - after_backward
            if should_track_epoch(ep, nepochs):
                backbone_tracked_total_sec += after_forward - start_time
                backward_tracked_total_sec += after_backward - after_loss
                optimizer_tracked_total_sec += after_optimizer - after_backward
            _dict_agg(epoch_stats, "train_loss", train_loss.detach().cpu().numpy())
            _dict_agg(epoch_stats, "train_time", after_optimizer - start_time, op="sum")
        train_epoch_elapsed = time.time() - train_epoch_t0
        train_epoch_time_total += train_epoch_elapsed
        if should_track_epoch(ep, nepochs):
            train_epoch_time_tracked += train_epoch_elapsed

        solver_net.eval()
        with torch.no_grad():
            val_phase_totals = {"backbone": 0.0, "projection": 0.0}
            val_epoch_t0 = time.time()
            for Xvalid, Yvalid in valid_loader:
                Xvalid = Xvalid.to(module.DEVICE)
                Yvalid = Yvalid.to(module.DEVICE)
                _eval_baseline_eq_nn_with_timing(module, data, Xvalid, Yvalid, solver_net, args, "valid", epoch_stats, val_phase_totals)
            val_epoch_elapsed = time.time() - val_epoch_t0
            val_epoch_time_total += val_epoch_elapsed
            backbone_total_sec += val_phase_totals["backbone"]
            projection_total_sec += val_phase_totals["projection"]
            if should_track_epoch(ep, nepochs):
                val_epoch_time_tracked += val_epoch_elapsed
                backbone_tracked_total_sec += val_phase_totals["backbone"]
                projection_tracked_total_sec += val_phase_totals["projection"]

            for Xtrain_eval, Ytrain_eval in train_eval_loader:
                Xtrain_eval = Xtrain_eval.to(module.DEVICE)
                Ytrain_eval = Ytrain_eval.to(module.DEVICE)
                _eval_baseline_eq_nn_with_timing(module, data, Xtrain_eval, Ytrain_eval, solver_net, args, "train", epoch_stats)
            for Xtest, Ytest in test_loader:
                Xtest = Xtest.to(module.DEVICE)
                Ytest = Ytest.to(module.DEVICE)
                _eval_baseline_eq_nn_with_timing(module, data, Xtest, Ytest, solver_net, args, "test", epoch_stats)

        if (ep % print_every) == 0 or ep == nepochs - 1:
            _save_epoch_summary(ep, epoch_stats)

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
            _save_model_artifacts(stats, solver_net, save_dir)

    _save_model_artifacts(stats, solver_net, save_dir)
    return solver_net, stats, {
        "training_wall_time_sec": time.time() - training_wall_t0,
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
        "batch_size": batch_size,
    }


def _predict_dc3(module, problem, solver_net, args, X):
    Y = solver_net(X)
    Ycorr, _ = module.grad_steps_all(problem, X, Y, args)
    return Ycorr


def _predict_dc3_raw(_module, _problem, solver_net, _args, X):
    return solver_net(X)


def _predict_baseline_nn(module, problem, solver_net, args, X):
    Y = solver_net(X)
    Ycorr, _ = module.grad_steps_all(problem, X, Y, args)
    return Ycorr


def _predict_baseline_nn_raw(_module, _problem, solver_net, _args, X):
    return solver_net(X)


def _predict_baseline_eq_nn(module, problem, solver_net, args, X):
    Z = solver_net(X)
    Y = module.complete_f(problem, X, Z, args)
    Ycorr, _ = module.grad_steps_all(problem, X, Y, args)
    return Ycorr


def _predict_baseline_eq_nn_raw(module, problem, solver_net, args, X):
    Z = solver_net(X)
    return module.complete_f(problem, X, Z, args)


def _prepare_poly_problem(case_dir: Path, data_cfg: dict, cfg_dict: dict):
    poly_torch._validate_data_cfg(data_cfg)
    if uses_nonconvex_poly_generator(data_cfg):
        generator = build_poly_problem_generator(data_cfg)
        problem_data_for_labels = dict(generator.get_problem_data())
        problem_data_for_labels["problem_type"] = normalize_poly_problem_type(str(data_cfg["type"]))
        label_model_def = build_poly_problem_model_from_data(problem_data_for_labels, dtype=jnp.float64)
        dataset = ensure_poly_dataset(
            case_dir,
            data_cfg,
            lambda: poly_nlpopt._generate_dataset_from_generator(generator, data_cfg),
            force=bool(data_cfg.get("force_regenerate", False)),
        )
    else:
        label_model_def = build_poly_problem_model(data_cfg, dtype=jnp.float64)
        dataset = ensure_poly_dataset(
            case_dir,
            data_cfg,
            lambda: poly_nlpopt._generate_dataset(label_model_def, data_cfg),
            force=bool(data_cfg.get("force_regenerate", False)),
        )
    train_frac, valid_frac, test_frac = poly_torch._resolve_split_fracs(cfg_dict)
    problem_data = poly_torch._build_problem_data(data_cfg)
    problem = poly_torch.QPQCQPDC3Problem(
        problem_data,
        dataset.X,
        dataset.Y,
        train_frac=train_frac,
        valid_frac=valid_frac,
        test_frac=test_frac,
        split_seed=int(cfg_dict.get("seed", 42)),
    )
    return dataset, problem


def _prepare_nlp_problem(case_dir: Path, data_cfg: dict, cfg_dict: dict):
    nlp_torch._validate_data_cfg(data_cfg)
    generator = nlp_torch._build_problem_generator(data_cfg)
    dataset = ensure_nlp_dataset(
        case_dir,
        data_cfg,
        lambda: nlp_torch._generate_dataset(generator, data_cfg),
        force=bool(data_cfg.get("force_regenerate", False)),
    )
    train_frac, valid_frac, test_frac = nlp_torch._resolve_split_fracs(cfg_dict)
    problem = nlp_torch.NLPDC3Problem(
        generator.get_problem_data(),
        dataset.X,
        dataset.Y,
        train_frac=train_frac,
        valid_frac=valid_frac,
        test_frac=test_frac,
        split_seed=int(cfg_dict.get("seed", 42)),
    )
    return dataset, problem


def _run_nlpopt(
    case_dir: Path,
    data_cfg: dict,
    cfg_dict: dict,
    proj_cfg: dict,
    *,
    output_dir: Path,
) -> RunArtifacts:
    clean_cfg = _training_cfg_only(cfg_dict)
    family = _problem_family(data_cfg)
    if family == "poly":
        poly_nlpopt.run_case(
            case_dir,
            data_cfg_override=data_cfg,
            cfg_dict_override=clean_cfg,
            proj_cfg_override=proj_cfg,
            output_dir_override=output_dir,
        )
        dataset_dir = poly_dataset_dir(case_dir, data_cfg)
    else:
        nlp_nlpopt.run_case(
            case_dir,
            data_cfg_override=data_cfg,
            cfg_dict_override=clean_cfg,
            proj_cfg_override=proj_cfg,
            output_dir_override=output_dir,
        )
        dataset_dir = nlp_dataset_dir(case_dir, data_cfg)
    _write_run_configs(output_dir, data_cfg, cfg_dict, proj_cfg)
    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        summary_payload = _load_json(summary_path)
        summary_payload["space_mb"] = _directory_size_mb(output_dir)
        summary_payload["dataset_dir"] = str(dataset_dir)
        summary_payload["save_dir"] = str(output_dir)
        _write_json(summary_path, summary_payload)
    return RunArtifacts(
        framework="nlpopt",
        dataset_dir=dataset_dir,
        run_dir=output_dir,
        history_path=output_dir / "run_history.json",
        metrics_path=summary_path,
        plot_path=output_dir / "compare_metrics.png",
    )


def _run_torch_framework(
    case_dir: Path,
    data_cfg: dict,
    cfg_dict: dict,
    proj_cfg: dict,
    framework: str,
    *,
    output_dir: Path,
) -> RunArtifacts:
    if torch is None:
        raise RuntimeError("Torch is required for DC3, Baseline NN, and EqNN runs.")

    seed_torch_runtime(int(cfg_dict.get("seed", 42)))
    family = _problem_family(data_cfg)
    dataset, problem = _prepare_nlp_problem(case_dir, data_cfg, cfg_dict) if family == "nlp" else _prepare_poly_problem(case_dir, data_cfg, cfg_dict)

    if framework == "dc3":
        module = nlp_torch._load_dc3_method_module() if family == "nlp" else poly_torch._load_dc3_method_module()
        args = nlp_torch._build_dc3_args(cfg_dict, use_completion=problem._use_completion) if family == "nlp" else poly_torch._build_dc3_args(cfg_dict, data_cfg, use_completion=problem._use_completion)
    elif framework == "baseline_nn":
        module = _load_dc3_runtime_module("baseline_nn.py", "_nlpopt_dc3_baseline_nn_runtime")
        args = _build_baseline_nn_args(cfg_dict)
    elif framework == "baseline_eq_nn":
        module = _load_dc3_runtime_module("baseline_eq_nn.py", "_nlpopt_dc3_baseline_eq_nn_runtime")
        args = _build_baseline_eq_nn_args(cfg_dict, use_completion=problem._use_completion)
    else:
        raise ValueError(f"Unsupported torch framework '{framework}'.")

    save_dir = Path(output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    problem.to(module.DEVICE)

    _write_json(save_dir / "args.json", args)
    with open(save_dir / "args.dict", "wb") as fh:
        pickle.dump(args, fh)
    _write_run_configs(save_dir, data_cfg, cfg_dict, proj_cfg)

    print(f"Dataset: {dataset.dataset_dir}")
    print(f"Dataset status: {'generated' if dataset.generated else 'reused'}")
    print(f"{_framework_label(framework)} save dir: {save_dir}")
    print(_problem_shape_text(data_cfg))
    print(f"Device: {module.DEVICE}")
    print(
        f"batch_size={int(args['batchSize'])}  "
        f"train_batches={_num_batches(problem.trainX.shape[0], int(args['batchSize']))}  "
        f"val_batches={_num_batches(problem.validX.shape[0], int(args['batchSize']))}  "
        f"test_batches={_num_batches(problem.testX.shape[0], int(args['batchSize']))}"
    )
    print(f"Splits: train={problem.trainX.shape[0]}  valid={problem.validX.shape[0]}  test={problem.testX.shape[0]}")

    if framework == "dc3":
        solver_net, stats, profile = poly_torch._train_net_with_nlpopt_logging(module, problem, args, str(save_dir))
        predict_fn = lambda X: _predict_dc3(module, problem, solver_net, args, X)
        raw_predict_fn = lambda X: _predict_dc3_raw(module, problem, solver_net, args, X)
    elif framework == "baseline_nn":
        solver_net, stats, profile = _train_baseline_nn_with_logging(module, problem, args, save_dir)
        predict_fn = lambda X: _predict_baseline_nn(module, problem, solver_net, args, X)
        raw_predict_fn = lambda X: _predict_baseline_nn_raw(module, problem, solver_net, args, X)
    else:
        solver_net, stats, profile = _train_baseline_eq_nn_with_logging(module, problem, args, save_dir)
        predict_fn = lambda X: _predict_baseline_eq_nn(module, problem, solver_net, args, X)
        raw_predict_fn = lambda X: _predict_baseline_eq_nn_raw(module, problem, solver_net, args, X)

    history_path = _save_history_and_plot(
        save_dir,
        framework,
        stats,
        problem,
        extra_history={
            **history_optimizer_timing_fields(dataset.metadata),
            "timing_start_epoch": int(summarize_timing_profile(profile)["timing_start_epoch"]),
            "timing_epochs_recorded": int(summarize_timing_profile(profile)["timing_epochs_recorded"]),
        },
    )
    metrics = _print_final_supervised_summary(problem, predict_fn, profile, predict_raw_fn=raw_predict_fn)
    summary = _summarize_stats(stats)
    summary.update(metrics)
    train_count = int(problem.trainX.shape[0])
    val_count = int(problem.validX.shape[0])
    summary["objective_value"] = _weighted_final_mean(stats, "train_eval", "valid_eval", train_count=train_count, val_count=val_count)
    summary["max_equality"] = _combined_final_max(stats, "train_eq_max", "valid_eq_max")
    summary["mean_equality"] = _weighted_final_mean(stats, "train_eq_mean", "valid_eq_mean", train_count=train_count, val_count=val_count)
    summary["max_inequality"] = _combined_final_max(stats, "train_ineq_max", "valid_ineq_max")
    summary["mean_inequality"] = _weighted_final_mean(stats, "train_ineq_mean", "valid_ineq_mean", train_count=train_count, val_count=val_count)
    summary["consistency"] = _weighted_final_mean(stats, "train_dist", "valid_dist", train_count=train_count, val_count=val_count)
    summary["optimality_gap"] = float(summary.get("relative_objective_gap", float("nan")))
    summary["dataset_dir"] = str(dataset.dataset_dir)
    summary["save_dir"] = str(save_dir)
    summary["framework"] = framework
    summary_path = save_dir / "summary.json"
    _write_json(summary_path, summary)
    summary["space_mb"] = _directory_size_mb(save_dir)
    _write_json(summary_path, summary)
    print(f"[{framework}] Saved: {save_dir}")

    return RunArtifacts(
        framework=framework,
        dataset_dir=dataset.dataset_dir,
        run_dir=save_dir,
        history_path=history_path,
        metrics_path=summary_path,
        plot_path=save_dir / "compare_metrics.png",
    )


def _run_single_case(
    case_dir: Path,
    data_cfg: dict,
    cfg_dict: dict,
    proj_cfg: dict,
    *,
    output_dir_override: Optional[Path] = None,
) -> RunArtifacts:
    data_cfg = _filtered_data_cfg(data_cfg)
    framework = _normalize_model_name(str(cfg_dict.get("model", "nlpopt")))
    _print_run_header(case_dir, data_cfg, cfg_dict, framework)
    dataset_dir = _resolve_dataset_dir(case_dir, data_cfg)
    output_dir = Path(output_dir_override) if output_dir_override is not None else _framework_dir(dataset_dir, framework)
    if framework == "nlpopt":
        return _run_nlpopt(case_dir, data_cfg, cfg_dict, proj_cfg, output_dir=output_dir)
    return _run_torch_framework(case_dir, data_cfg, cfg_dict, proj_cfg, framework, output_dir=output_dir)


def _resolve_dataset_dir(case_dir: Path, data_cfg: dict) -> Path:
    data_cfg = _filtered_data_cfg(data_cfg)
    return nlp_dataset_dir(case_dir, data_cfg) if _problem_family(data_cfg) == "nlp" else poly_dataset_dir(case_dir, data_cfg)


def _run_multi_seed(
    case_dir: Path,
    data_cfg: dict,
    cfg_dict: dict,
    proj_cfg: dict,
    *,
    multi_dir_override: Optional[Path] = None,
) -> int:
    data_cfg = _filtered_data_cfg(data_cfg)
    framework = _normalize_model_name(str(cfg_dict.get("model", "nlpopt")))
    base_seed = int(cfg_dict.get("seed", 42))
    num_seeds = int(cfg_dict.get("num_seeds", 10))
    if num_seeds <= 0:
        raise ValueError("num_seeds must be positive when run_multiple_seed is true.")
    seeds = [base_seed + idx for idx in range(num_seeds)]

    dataset_dir = _resolve_dataset_dir(case_dir, data_cfg)
    multi_dir = Path(multi_dir_override) if multi_dir_override is not None else _framework_multi_dir(dataset_dir, framework)
    multi_dir.mkdir(parents=True, exist_ok=True)
    _write_run_configs(multi_dir, data_cfg, cfg_dict, proj_cfg)

    run_manifests = []
    history_payloads = []

    for seed in seeds:
        run_cfg = copy.deepcopy(cfg_dict)
        run_cfg["seed"] = int(seed)
        run_cfg["run_multiple_seed"] = False
        print("")
        print(f"[multi-seed] Running {framework} with config.seed={seed}")
        seed_dir = multi_dir / str(int(seed))
        artifacts = _run_single_case(case_dir, data_cfg, run_cfg, proj_cfg, output_dir_override=seed_dir)

        manifest = {
            "seed": int(seed),
            "framework": framework,
            "dataset_dir": str(artifacts.dataset_dir),
            "run_dir": str(artifacts.run_dir),
            "history_path": str(artifacts.history_path),
            "metrics_path": str(artifacts.metrics_path),
            "plot_path": str(artifacts.plot_path),
            "seed_dir": str(seed_dir),
        }
        _write_json(seed_dir / "manifest.json", manifest)
        run_manifests.append(manifest)
        history_payloads.append(_load_json(artifacts.history_path))

    if history_payloads:
        epochs = history_payloads[0]["epochs"]
        for payload in history_payloads[1:]:
            if payload["epochs"] != epochs:
                raise ValueError("All runs must share the same epoch checkpoints for shadow plotting.")
        shadow_plot_path = multi_dir / "compare_metrics_shadow.png"
        save_shadow_objective_violation_plot(
            shadow_plot_path,
            epochs=epochs,
            train_gap_pct_runs=[payload["train_worst_relative_gap_pct"] for payload in history_payloads],
            val_gap_pct_runs=[payload["val_worst_relative_gap_pct"] for payload in history_payloads],
            train_violation_runs=[payload["train_violation"] for payload in history_payloads],
            val_violation_runs=[payload["val_violation"] for payload in history_payloads],
            series_label=_framework_label(framework),
        )
    else:
        shadow_plot_path = None

    summary_path = multi_dir / "multi_seed_summary.json"
    _write_json(
        summary_path,
        {
            "framework": framework,
            "case_dir": str(case_dir),
            "dataset_dir": str(dataset_dir),
            "multi_dir": str(multi_dir),
            "seeds": seeds,
            "shadow_plot_path": str(shadow_plot_path) if shadow_plot_path is not None else None,
            "runs": run_manifests,
        },
    )
    metadata_path = _append_family_metadata(
        dataset_dir,
        mode="multi_seed_model",
        output_dir=multi_dir,
        data_cfg=data_cfg,
        cfg_dict=cfg_dict,
        proj_cfg=proj_cfg,
        framework=framework,
        seeds=seeds,
        extra={
            "summary_path": str(summary_path),
            "shadow_plot_path": str(shadow_plot_path) if shadow_plot_path is not None else None,
        },
    )
    print("")
    if shadow_plot_path is not None:
        print(f"[multi-seed] Saved shadow plot: {shadow_plot_path}")
    print(f"[multi-seed] Saved summary: {summary_path}")
    print(f"[multi-seed] Updated family metadata: {metadata_path}")
    return 0


def run_root_config() -> int:
    case_dir = _case_workspace()
    data_path = ROOT / "data.json"
    cfg_path = ROOT / "config.json"
    proj_path = ROOT / "proj.json"
    if not data_path.exists() or not cfg_path.exists() or not proj_path.exists():
        raise FileNotFoundError(
            "Expected root-level data.json, config.json, and proj.json next to PIPELINE.md. "
            "Create those files first, then rerun main.py with the appropriate --type."
        )

    data_cfg = _filtered_data_cfg(_load_json(data_path))
    cfg_dict = _load_json(cfg_path)
    proj_cfg = _load_json(proj_path)

    if _str_to_bool(cfg_dict.get("run_multiple_seed", False)):
        return _run_multi_seed(case_dir, data_cfg, cfg_dict, proj_cfg)

    artifacts = _run_single_case(case_dir, data_cfg, cfg_dict, proj_cfg)
    metadata_path = _append_family_metadata(
        artifacts.dataset_dir,
        mode="single_model",
        output_dir=artifacts.run_dir,
        data_cfg=data_cfg,
        cfg_dict=cfg_dict,
        proj_cfg=proj_cfg,
        framework=artifacts.framework,
        seeds=[int(cfg_dict.get("seed", 42))],
        extra={"summary_path": str(artifacts.metrics_path), "history_path": str(artifacts.history_path)},
    )
    print(f"[run] Updated family metadata: {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_root_config())
