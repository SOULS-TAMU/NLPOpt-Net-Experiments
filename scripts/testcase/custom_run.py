#!/usr/bin/env python3
from __future__ import annotations

import copy
import csv
import hashlib
import json
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Optional

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "nlpopt" / "src"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scripts.plot_utils.plotting import save_shadow_objective_violation_plot  # noqa: E402
from scripts.misc.cli_overrides import apply_cli_overrides  # noqa: E402
from scripts.testcase import unified_runner as unified  # noqa: E402
from scripts.testcase import nlp_run as nlp_nlpopt  # noqa: E402
from scripts.misc.optimizer_profile import enrich_optimizer_generation_metadata  # noqa: E402
from scripts.misc.inequality_multipliers import coerce_ineq_multipliers  # noqa: E402
from case.custom.cases import (  # noqa: E402
    DATASET_KIND,
    build_problem_generator,
    build_problem_model_from_data,
    normalize_problem_type,
    parse_problem_spec,
    problem_shape_from_data_cfg,
)

jax.config.update("jax_enable_x64", True)

SCHEMA_VERSION = 2
_LOCAL_REQUIRED_KEYS = ("type", "num_samples", "seed", "solver", "x_L", "x_U")


@dataclass(frozen=True)
class DatasetBundle:
    dataset_dir: Path
    dataset_id: str
    generated: bool
    X: np.ndarray
    Y: np.ndarray
    Mu: np.ndarray
    metadata: dict[str, Any]


def _case_workspace() -> Path:
    case_dir = ROOT / "case" / "custom"
    case_dir.mkdir(parents=True, exist_ok=True)
    return case_dir


def _local_paths() -> tuple[Path, Path, Path, Path]:
    case_dir = _case_workspace()
    return (
        case_dir / "data.json",
        case_dir / "config.json",
        case_dir / "proj.json",
        case_dir / "problem.json",
    )


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)


def _load_local_configs() -> tuple[dict, dict, dict, dict]:
    data_path, cfg_path, proj_path, problem_path = _local_paths()
    missing = [path.name for path in (data_path, cfg_path, proj_path, problem_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Expected local files in case/custom/: " + ", ".join(missing)
        )
    data_cfg = _load_json(data_path)
    cfg_dict = _load_json(cfg_path)
    proj_cfg = _load_json(proj_path)
    problem_cfg = _load_json(problem_path)
    data_cfg, cfg_dict = apply_cli_overrides(data_cfg, cfg_dict)
    return data_cfg, cfg_dict, proj_cfg, problem_cfg


def _json_hash(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:12]


def _normalize_problem_cfg(problem_cfg: Mapping[str, Any]) -> dict[str, Any]:
    parsed = parse_problem_spec(problem_cfg)
    return {
        "objective": parsed.objective_text,
        "constraints": [constraint.original for constraint in parsed.constraints],
    }


def _normalize_local_data_cfg(data_cfg: Mapping[str, Any], problem_cfg: Mapping[str, Any]) -> dict[str, Any]:
    missing = [key for key in _LOCAL_REQUIRED_KEYS if key not in data_cfg]
    if missing:
        raise ValueError(f"case/custom/data.json is missing required keys: {', '.join(sorted(missing))}")

    normalize_problem_type(str(data_cfg["type"]))
    parsed = parse_problem_spec(problem_cfg)
    x_l = np.asarray(data_cfg["x_L"], dtype=np.float64)
    x_u = np.asarray(data_cfg["x_U"], dtype=np.float64)
    if x_l.shape != (parsed.n_x,) or x_u.shape != (parsed.n_x,):
        raise ValueError(
            f"x_L and x_U must both have shape ({parsed.n_x},) for the parameters used in problem.json."
        )
    solver = str(data_cfg.get("solver", "auto")).strip().lower()
    if solver not in {"auto", "scs", "cvxpy", "gurobi"}:
        raise ValueError("solver must be one of: auto, scs, cvxpy, gurobi.")

    return {
        "type": DATASET_KIND,
        "num_samples": int(data_cfg["num_samples"]),
        "seed": int(data_cfg["seed"]),
        "solver": solver,
        "x_L": x_l.astype(float).tolist(),
        "x_U": x_u.astype(float).tolist(),
        "schema_version": int(data_cfg.get("schema_version", SCHEMA_VERSION)),
    }


def _canonical_data_cfg(data_cfg: Mapping[str, Any], problem_cfg: Mapping[str, Any]) -> dict[str, Any]:
    if "problem_spec" in data_cfg and "N_points" in data_cfg:
        local = {
            "type": DATASET_KIND,
            "num_samples": int(data_cfg["N_points"]),
            "seed": int(data_cfg["seed"]),
            "solver": str(data_cfg.get("solver", "auto")),
            "x_L": copy.deepcopy(list(data_cfg["x_L"])),
            "x_U": copy.deepcopy(list(data_cfg["x_U"])),
            "schema_version": int(data_cfg.get("schema_version", SCHEMA_VERSION)),
        }
    else:
        local = _normalize_local_data_cfg(data_cfg, problem_cfg)
    return {
        "type": local["type"],
        "num_samples": local["num_samples"],
        "seed": local["seed"],
        "solver": local["solver"],
        "x_L": local["x_L"],
        "x_U": local["x_U"],
        "schema_version": local["schema_version"],
        "problem_hash": _json_hash(_normalize_problem_cfg(problem_cfg)),
    }


def _translated_custom_cfg(data_cfg: Mapping[str, Any], problem_cfg: Mapping[str, Any]) -> dict[str, Any]:
    local = _normalize_local_data_cfg(data_cfg, problem_cfg)
    parsed = parse_problem_spec(problem_cfg)
    return {
        "type": DATASET_KIND,
        "n_x": parsed.n_x,
        "n_y": parsed.n_y,
        "n_eq": parsed.n_eq,
        "n_ineq": parsed.n_ineq,
        "N_samples": int(local["num_samples"]),
        "N_points": int(local["num_samples"]),
        "seed": int(local["seed"]),
        "x_L": copy.deepcopy(local["x_L"]),
        "x_U": copy.deepcopy(local["x_U"]),
        "solver": str(local["solver"]),
        "problem_spec": _normalize_problem_cfg(problem_cfg),
        "force_regenerate": bool(local.get("force_regenerate", False)),
    }


def _coerce_translated_cfg(data_cfg: Mapping[str, Any], problem_cfg: Mapping[str, Any]) -> dict[str, Any]:
    if "problem_spec" in data_cfg and "N_points" in data_cfg and "N_samples" in data_cfg:
        return copy.deepcopy(dict(data_cfg))
    return _translated_custom_cfg(data_cfg, problem_cfg)


def build_dataset_id(data_cfg: Mapping[str, Any], problem_cfg: Mapping[str, Any]) -> str:
    translated = _coerce_translated_cfg(data_cfg, problem_cfg)
    problem_hash = _json_hash(_normalize_problem_cfg(problem_cfg))
    stem = (
        f"custom_{'cvx' if parse_problem_spec(problem_cfg).is_convex else 'noncvx'}_"
        f"nx{translated['n_x']}_ny{translated['n_y']}_"
        f"neq{translated['n_eq']}_nineq{translated['n_ineq']}_"
        f"ns{translated['N_points']}_seed{translated['seed']}"
    )
    return f"{stem}_{problem_hash}"


def dataset_dir(case_dir: Path, data_cfg: Mapping[str, Any], problem_cfg: Mapping[str, Any]) -> Path:
    return case_dir / "problem_data" / DATASET_KIND / build_dataset_id(data_cfg, problem_cfg)


def _paths(base: Path) -> dict[str, Path]:
    return {
        "arrays": base / "dataset.npz",
        "parameters_csv": base / "parameters.csv",
        "variables_csv": base / "variables.csv",
        "ineq_multipliers_csv": base / "ineq_multipliers.csv",
        "metadata": base / "metadata.json",
        "data": base / "data.json",
        "problem": base / "problem.json",
        "problem_data": base / "problem_data.npz",
    }


def _write_csv(path: Path, arr: np.ndarray) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerows(np.asarray(arr).tolist())


def _dataset_exists(base: Path) -> bool:
    paths = _paths(base)
    return all(path.exists() for path in paths.values())


def _save_dataset(
    base: Path,
    *,
    data_cfg: Mapping[str, Any],
    problem_cfg: Mapping[str, Any],
    X: np.ndarray,
    Y: np.ndarray,
    Mu: np.ndarray,
    metadata: Mapping[str, Any],
    problem_data: Mapping[str, Any],
) -> dict[str, Any]:
    base.mkdir(parents=True, exist_ok=True)
    paths = _paths(base)
    np.savez(paths["arrays"], X=np.asarray(X, dtype=np.float64), Y=np.asarray(Y, dtype=np.float64), Mu=np.asarray(Mu, dtype=np.float64))
    _write_csv(paths["parameters_csv"], X)
    _write_csv(paths["variables_csv"], Y)
    _write_csv(paths["ineq_multipliers_csv"], Mu)
    _write_json(paths["data"], _canonical_data_cfg(data_cfg, problem_cfg))
    _write_json(paths["problem"], _normalize_problem_cfg(problem_cfg))
    np.savez(paths["problem_data"], **{key: np.asarray(value) for key, value in dict(problem_data).items()})
    enriched_metadata = enrich_optimizer_generation_metadata(
        metadata,
        num_points=int(np.asarray(X).shape[0]),
        artifact_paths=(
            paths["arrays"],
            paths["parameters_csv"],
            paths["variables_csv"],
            paths["ineq_multipliers_csv"],
            paths["data"],
            paths["problem"],
            paths["problem_data"],
        ),
    )
    _write_json(paths["metadata"], enriched_metadata)
    return enriched_metadata


def _load_dataset(base: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    paths = _paths(base)
    arrays = np.load(paths["arrays"])
    metadata = _load_json(paths["metadata"])
    return np.asarray(arrays["X"]), np.asarray(arrays["Y"]), np.asarray(arrays["Mu"]), metadata


def ensure_cached_dataset(
    case_dir: Path,
    data_cfg: Mapping[str, Any],
    generate_fn: Callable[[], tuple[np.ndarray, np.ndarray, dict[str, Any], Mapping[str, Any]]],
    *,
    force: bool = False,
) -> DatasetBundle:
    if "problem_spec" not in data_cfg:
        raise ValueError("Custom dataset cache requires a translated data config with embedded problem_spec.")
    translated = copy.deepcopy(dict(data_cfg))
    problem_cfg = translated["problem_spec"]
    base = dataset_dir(case_dir, translated, problem_cfg)
    dataset_id = build_dataset_id(translated, problem_cfg)
    if _dataset_exists(base) and not force:
        X, Y, Mu, metadata = _load_dataset(base)
        return DatasetBundle(dataset_dir=base, dataset_id=dataset_id, generated=False, X=X, Y=Y, Mu=Mu, metadata=metadata)

    X, Y, Mu, metadata, problem_data = generate_fn()
    metadata = _save_dataset(
        base,
        data_cfg=translated,
        problem_cfg=problem_cfg,
        X=X,
        Y=Y,
        Mu=Mu,
        metadata=metadata,
        problem_data=problem_data,
    )
    return DatasetBundle(dataset_dir=base, dataset_id=dataset_id, generated=True, X=np.asarray(X), Y=np.asarray(Y), Mu=np.asarray(Mu), metadata=dict(metadata))


def _generate_dataset(generator, data_cfg: Mapping[str, Any]):
    start_time = time.perf_counter()
    n_samples = int(data_cfg["N_samples"])
    target_points = int(data_cfg["N_points"])
    max_attempts = max(target_points * 10, target_points + 100)
    kept_x = []
    kept_y = []
    kept_mu = []
    objectives = []
    status_counts: dict[str, int] = {}

    xs = generator.sample_parameters(max_attempts)
    for x_value in xs:
        result = generator.solve_for_x(x_value)
        status = str(result["status"])
        status_counts[status] = status_counts.get(status, 0) + 1
        if status in ("optimal", "optimal_inaccurate") and result["y"] is not None:
            kept_x.append(np.asarray(x_value, dtype=np.float64))
            kept_y.append(np.asarray(result["y"], dtype=np.float64))
            kept_mu.append(coerce_ineq_multipliers(result.get("mu"), generator.n_ineq))
            objectives.append(float(result["objective"]) if result["objective"] is not None else np.nan)
            if len(kept_x) >= target_points:
                break

    if len(kept_x) < target_points:
        raise RuntimeError(
            f"Only collected {len(kept_x)} successful custom-problem points out of requested {target_points}. "
            "Increase the feasible region or dataset sampling budget."
        )

    X = np.asarray(kept_x, dtype=np.float64)
    Y = np.asarray(kept_y, dtype=np.float64)
    metadata = {
        "problem_type": DATASET_KIND,
        "n_x": int(generator.n_x),
        "n_y": int(generator.n_y),
        "n_eq": int(generator.n_eq),
        "n_ineq": int(generator.n_ineq),
        "N_samples": n_samples,
        "N_points": target_points,
        "solver_backend": str(generator.solver),
        "is_convex": bool(generator.is_convex),
        "convexity_reason": str(generator.parsed.convexity_reason),
        "seed": int(data_cfg["seed"]),
        "objective_min": float(np.nanmin(objectives)) if objectives else np.nan,
        "objective_max": float(np.nanmax(objectives)) if objectives else np.nan,
        "objective_mean": float(np.nanmean(objectives)) if objectives else np.nan,
        "status_counts": status_counts,
        "optimizer_generation_wall_time_sec": time.perf_counter() - start_time,
    }
    Mu = np.stack(kept_mu, axis=0) if kept_mu else np.zeros((X.shape[0], generator.n_ineq), dtype=np.float64)
    return X, Y, Mu, metadata, generator.get_problem_data()


def _problem_shape_text(problem_cfg: Mapping[str, Any], data_cfg: Mapping[str, Any]) -> str:
    translated = _translated_custom_cfg(data_cfg, problem_cfg)
    shape = problem_shape_from_data_cfg(translated)
    return (
        f"CUSTOM  n_x={shape['n_x']} n_y={shape['n_y']} "
        f"n_eq={shape['n_eq']} n_ineq={shape['n_ineq']} "
        f"({'convex' if shape['is_convex'] else 'nonconvex'} via {shape['solver_backend']})"
    )


def _validate_translated_data_cfg(data_cfg: Mapping[str, Any]) -> None:
    if str(data_cfg["type"]).strip().lower() != DATASET_KIND:
        raise ValueError("Unsupported custom testcase type.")
    for key in ("n_x", "n_y", "n_eq", "n_ineq", "N_samples", "N_points", "seed", "x_L", "x_U", "problem_spec"):
        if key not in data_cfg:
            raise ValueError(f"Missing required translated custom field '{key}'.")


def _write_local_run_configs(run_dir: Path, data_cfg: dict, cfg_dict: dict, proj_cfg: dict, problem_cfg: dict) -> None:
    unified._write_run_configs(run_dir, data_cfg, cfg_dict, proj_cfg)
    _write_json(run_dir / "problem.json", _normalize_problem_cfg(problem_cfg))


@contextmanager
def _patch_nlp_nlpopt_for_custom():
    original_validate_data_cfg = nlp_nlpopt._validate_data_cfg
    original_build_problem_generator = nlp_nlpopt.build_problem_generator
    original_build_problem_model_from_data = nlp_nlpopt.build_problem_model_from_data
    original_normalize_problem_type = nlp_nlpopt.normalize_problem_type
    original_ensure_cached_dataset = nlp_nlpopt.ensure_cached_dataset
    original_generate_dataset = nlp_nlpopt._generate_dataset

    nlp_nlpopt._validate_data_cfg = _validate_translated_data_cfg
    nlp_nlpopt.build_problem_generator = build_problem_generator
    nlp_nlpopt.build_problem_model_from_data = lambda problem_data, dtype=jnp.float64: build_problem_model_from_data(problem_data, dtype=dtype)
    nlp_nlpopt.normalize_problem_type = normalize_problem_type
    nlp_nlpopt.ensure_cached_dataset = ensure_cached_dataset
    nlp_nlpopt._generate_dataset = _generate_dataset
    try:
        yield
    finally:
        nlp_nlpopt._validate_data_cfg = original_validate_data_cfg
        nlp_nlpopt.build_problem_generator = original_build_problem_generator
        nlp_nlpopt.build_problem_model_from_data = original_build_problem_model_from_data
        nlp_nlpopt.normalize_problem_type = original_normalize_problem_type
        nlp_nlpopt.ensure_cached_dataset = original_ensure_cached_dataset
        nlp_nlpopt._generate_dataset = original_generate_dataset


def _run_nlpopt(
    case_dir: Path,
    data_cfg: dict,
    cfg_dict: dict,
    proj_cfg: dict,
    problem_cfg: dict,
    *,
    output_dir: Path,
) -> unified.RunArtifacts:
    translated_cfg = _translated_custom_cfg(data_cfg, problem_cfg)
    with _patch_nlp_nlpopt_for_custom():
        nlp_nlpopt.run_case(
            case_dir,
            data_cfg_override=translated_cfg,
            cfg_dict_override=unified._training_cfg_only(cfg_dict),
            proj_cfg_override=proj_cfg,
            output_dir_override=output_dir,
        )
    dataset_root = dataset_dir(case_dir, data_cfg, problem_cfg)
    _write_local_run_configs(output_dir, _normalize_local_data_cfg(data_cfg, problem_cfg), cfg_dict, proj_cfg, problem_cfg)
    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        summary_payload = _load_json(summary_path)
        summary_payload["space_mb"] = unified._directory_size_mb(output_dir)
        summary_payload["dataset_dir"] = str(dataset_root)
        summary_payload["save_dir"] = str(output_dir)
        _write_json(summary_path, summary_payload)
    return unified.RunArtifacts(
        framework="nlpopt",
        dataset_dir=dataset_root,
        run_dir=output_dir,
        history_path=output_dir / "run_history.json",
        metrics_path=summary_path,
        plot_path=output_dir / "compare_metrics.png",
    )


def _print_run_header(case_dir: Path, data_cfg: dict, cfg_dict: dict, problem_cfg: dict) -> None:
    dataset_target = dataset_dir(case_dir, data_cfg, problem_cfg)
    print("=" * 80)
    print("Standalone runner | framework=NLPOpt")
    print(_problem_shape_text(problem_cfg, data_cfg))
    print(f"Workspace: {case_dir}")
    print(f"Dataset target: {dataset_target}")
    print(
        f"Config: seed={int(cfg_dict.get('seed', 42))} "
        f"epochs={int(cfg_dict.get('epochs', 1000))} "
        f"batch_size={int(cfg_dict.get('batch_size', 200))} "
        f"lr={float(cfg_dict.get('learning_rate', 1e-4)):.3e}"
    )
    print("=" * 80)


def _run_single_case(
    case_dir: Path,
    data_cfg: dict,
    cfg_dict: dict,
    proj_cfg: dict,
    problem_cfg: dict,
    *,
    output_dir_override: Optional[Path] = None,
) -> unified.RunArtifacts:
    framework = unified._normalize_model_name(str(cfg_dict.get("model", "nlpopt")))
    if framework != "nlpopt":
        raise ValueError("case/custom currently supports model='nlpopt' only.")
    _print_run_header(case_dir, data_cfg, cfg_dict, problem_cfg)
    dataset_root = dataset_dir(case_dir, data_cfg, problem_cfg)
    output_dir = Path(output_dir_override) if output_dir_override is not None else unified._framework_dir(dataset_root, framework)
    return _run_nlpopt(case_dir, data_cfg, cfg_dict, proj_cfg, problem_cfg, output_dir=output_dir)


def _run_multi_seed(case_dir: Path, data_cfg: dict, cfg_dict: dict, proj_cfg: dict, problem_cfg: dict) -> int:
    framework = "nlpopt"
    base_seed = int(cfg_dict.get("seed", 42))
    num_seeds = int(cfg_dict.get("num_seeds", 10))
    if num_seeds <= 0:
        raise ValueError("num_seeds must be positive when run_multiple_seed is true.")
    seeds = [base_seed + idx for idx in range(num_seeds)]

    dataset_root = dataset_dir(case_dir, data_cfg, problem_cfg)
    multi_dir = unified._framework_multi_dir(dataset_root, framework)
    multi_dir.mkdir(parents=True, exist_ok=True)
    _write_local_run_configs(multi_dir, _normalize_local_data_cfg(data_cfg, problem_cfg), cfg_dict, proj_cfg, problem_cfg)

    run_manifests = []
    history_payloads = []

    for seed in seeds:
        run_cfg = copy.deepcopy(cfg_dict)
        run_cfg["seed"] = int(seed)
        run_cfg["run_multiple_seed"] = False
        print("")
        print(f"[multi-seed] Running nlpopt with config.seed={seed}")
        seed_dir = unified._framework_seed_dir(dataset_root, framework, int(seed))
        artifacts = _run_single_case(case_dir, data_cfg, run_cfg, proj_cfg, problem_cfg, output_dir_override=seed_dir)

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

    shadow_plot_path = None
    if history_payloads:
        epochs = history_payloads[0]["epochs"]
        shadow_plot_path = multi_dir / "compare_metrics_shadow.png"
        save_shadow_objective_violation_plot(
            shadow_plot_path,
            epochs=epochs,
            train_gap_pct_runs=[payload["train_worst_relative_gap_pct"] for payload in history_payloads],
            val_gap_pct_runs=[payload["val_worst_relative_gap_pct"] for payload in history_payloads],
            train_violation_runs=[payload["train_violation"] for payload in history_payloads],
            val_violation_runs=[payload["val_violation"] for payload in history_payloads],
            series_label="NLPOpt",
        )

    summary_path = multi_dir / "multi_seed_summary.json"
    _write_json(
        summary_path,
        {
            "framework": framework,
            "framework_label": "NLPOpt",
            "dataset_dir": str(dataset_root),
            "multi_dir": str(multi_dir),
            "seeds": seeds,
            "runs": run_manifests,
            "shadow_plot_path": str(shadow_plot_path) if shadow_plot_path is not None else None,
        },
    )
    metadata_path = unified._append_family_metadata(
        dataset_root,
        mode="multi_seed",
        output_dir=multi_dir,
        data_cfg=_normalize_local_data_cfg(data_cfg, problem_cfg),
        cfg_dict=cfg_dict,
        proj_cfg=proj_cfg,
        framework=framework,
        seeds=seeds,
        extra={"summary_path": str(summary_path), "problem": _normalize_problem_cfg(problem_cfg)},
    )
    print(f"[multi-seed] Saved summary: {summary_path}")
    print(f"[multi-seed] Updated family metadata: {metadata_path}")
    return 0


def default_dataset_dir() -> Path:
    data_cfg, _cfg, _proj, problem_cfg = _load_local_configs()
    return dataset_dir(_case_workspace(), data_cfg, problem_cfg)


def run_case(_case_dir: Path | None = None, _path_arg: str | None = None) -> int:
    return main()


def main() -> int:
    data_cfg, cfg_dict, proj_cfg, problem_cfg = _load_local_configs()
    data_cfg = _normalize_local_data_cfg(data_cfg, problem_cfg)
    problem_cfg = _normalize_problem_cfg(problem_cfg)
    case_dir = _case_workspace()

    if unified._str_to_bool(cfg_dict.get("run_multiple_seed", False)):
        return _run_multi_seed(case_dir, data_cfg, cfg_dict, proj_cfg, problem_cfg)

    artifacts = _run_single_case(case_dir, data_cfg, cfg_dict, proj_cfg, problem_cfg)
    metadata_path = unified._append_family_metadata(
        artifacts.dataset_dir,
        mode="single_model",
        output_dir=artifacts.run_dir,
        data_cfg=data_cfg,
        cfg_dict=cfg_dict,
        proj_cfg=proj_cfg,
        framework=artifacts.framework,
        seeds=[int(cfg_dict.get("seed", 42))],
        extra={
            "summary_path": str(artifacts.metrics_path),
            "history_path": str(artifacts.history_path),
            "problem": problem_cfg,
        },
    )
    print(f"[run] Updated family metadata: {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
