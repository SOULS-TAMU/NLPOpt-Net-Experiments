#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.plot_utils.plotting import save_comparison_shadow_plot  # noqa: E402
from scripts.testcase import unified_runner as unified  # noqa: E402

_TABLE_ROWS = (
    ("optimizer", "Optimizer"),
    ("baseline_nn", "Baseline NN"),
    ("baseline_eq_nn", "Baseline Eq. NN"),
    ("dc3", "DC3"),
    ("nlpopt", "NLPOpt"),
)
_TABLE_ROW_LABELS = dict(_TABLE_ROWS)
_TABLE_COLUMNS = (
    ("objective_value", "Obj Value"),
    ("max_equality", "Max. Equality"),
    ("mean_equality", "Mean Equality"),
    ("max_inequality", "Max. Inequality"),
    ("mean_inequality", "Mean. Inequality"),
    ("consistency", "Consistency"),
    ("optimality_gap", "Optimality Gap"),
)
_RESOURCE_COLUMNS = (
    ("time_s_per_batch", "Time (s/batch)"),
    ("time_s_per_sample", "Time (s/sample)"),
    ("space_mb_per_batch", "Space (MB/batch)"),
    ("space_mb_per_sample", "Space (MB/sample)"),
)


def _comparison_dir(dataset_dir: Path) -> Path:
    direct_dir = dataset_dir / "comparison"
    if (direct_dir / "comparison_summary.json").exists():
        return direct_dir
    dc3_dir = dataset_dir / "comparison_dc3"
    if (dc3_dir / "comparison_summary.json").exists():
        return dc3_dir
    dc3_candidates = sorted(
        child for child in dc3_dir.iterdir()
        if child.is_dir() and (child / "comparison_summary.json").exists()
    ) if dc3_dir.exists() else []
    if len(dc3_candidates) == 1:
        return dc3_candidates[0]
    legacy_dir = dataset_dir / "comparison" / "nlpopt_vs_dc3"
    if (legacy_dir / "comparison_summary.json").exists():
        return legacy_dir
    return direct_dir


def _ordered_table_framework_rows(comparison_summary: dict) -> list[tuple[str, str]]:
    available = list(comparison_summary.get("frameworks", {}).keys())
    ordered = [("optimizer", _TABLE_ROW_LABELS["optimizer"])]
    for framework_key, row_label in _TABLE_ROWS:
        if framework_key == "optimizer":
            continue
        if framework_key in available:
            ordered.append((framework_key, row_label))
    for framework_key in available:
        if framework_key not in {key for key, _label in ordered}:
            ordered.append((framework_key, unified._framework_label(framework_key)))
    return ordered


def _resolve_dataset_or_comparison_dir(path_arg: str | None) -> tuple[Path, Path]:
    if path_arg is None:
        case_dir = unified._case_workspace()
        data_cfg = unified._filtered_data_cfg(unified._load_json(ROOT / "data.json"))
        dataset_dir = unified._resolve_dataset_dir(case_dir, data_cfg)
        return dataset_dir, _comparison_dir(dataset_dir)

    supplied = Path(path_arg).expanduser().resolve()
    if not supplied.exists():
        raise FileNotFoundError(f"Provided path does not exist: {supplied}")

    if supplied.is_dir() and (supplied / "comparison_summary.json").exists():
        comparison_dir = supplied
        if comparison_dir.name == "comparison_dc3":
            dataset_dir = comparison_dir.parent
        elif comparison_dir.parent.name == "comparison_dc3":
            dataset_dir = comparison_dir.parent.parent
        elif comparison_dir.parent.name == "comparison" and comparison_dir.name != "comparison":
            dataset_dir = comparison_dir.parent.parent
        else:
            dataset_dir = comparison_dir.parent
        return dataset_dir, comparison_dir

    if supplied.is_dir():
        comparison_dir = _comparison_dir(supplied)
        return supplied, comparison_dir

    raise NotADirectoryError(f"Expected a dataset or comparison directory, got: {supplied}")


def _load_framework_histories(summary_payload: dict) -> tuple[list[int], dict]:
    epochs = None
    model_runs = {}
    frameworks = list(summary_payload.get("frameworks", {}).keys())
    if not frameworks:
        raise ValueError("No frameworks found in comparison summary.")

    for framework in frameworks:
        framework_info = summary_payload["frameworks"][framework]
        multi_seed_summary = unified._load_json(Path(framework_info["summary_path"]))
        history_payloads = []
        for run in multi_seed_summary.get("runs", []):
            history_path = Path(run["history_path"])
            if not history_path.exists():
                raise FileNotFoundError(f"Missing run history at {history_path}")
            history_payloads.append(unified._load_json(history_path))
        if not history_payloads:
            raise ValueError(f"No run histories found for framework '{framework}'.")

        model_epochs = history_payloads[0]["epochs"]
        for payload in history_payloads[1:]:
            if payload["epochs"] != model_epochs:
                raise ValueError(f"Inconsistent epoch checkpoints within framework '{framework}'.")
        if epochs is None:
            epochs = model_epochs
        elif epochs != model_epochs:
            raise ValueError("Framework comparison requires matching epoch checkpoints across models.")

        model_runs[framework_info["label"]] = {
            "train_gap_pct_runs": [payload["train_worst_relative_gap_pct"] for payload in history_payloads],
            "val_gap_pct_runs": [payload["val_worst_relative_gap_pct"] for payload in history_payloads],
            "train_violation_runs": [payload["train_violation"] for payload in history_payloads],
            "val_violation_runs": [payload["val_violation"] for payload in history_payloads],
        }

    if epochs is None:
        raise ValueError("No epochs available for comparison plotting.")
    return epochs, model_runs


def _metric_value(payload: dict, key: str, run: dict) -> float | None:
    if key in payload and payload[key] is not None:
        return float(payload[key])
    if key == "optimality_gap" and payload.get("relative_objective_gap") is not None:
        return float(payload["relative_objective_gap"])
    if key == "max_equality" and payload.get("eq_inf") is not None:
        return float(payload["eq_inf"])
    if key == "mean_equality" and payload.get("eq_inf") is not None:
        return float(payload["eq_inf"])
    if key == "max_inequality" and payload.get("ineq_inf") is not None:
        return float(payload["ineq_inf"])
    if key == "mean_inequality" and payload.get("ineq_inf") is not None:
        return float(payload["ineq_inf"])
    if key == "space_mb":
        run_dir_value = run.get("run_dir")
        if run_dir_value:
            run_dir = Path(run_dir_value)
            if run_dir.exists():
                return unified._directory_size_mb(run_dir)
    return None


def _aggregate_values(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": None, "variance_pct": None, "display": "N/A"}
    mean = float(np.mean(arr))
    variance_pct = float(100.0 * np.std(arr) / max(abs(mean), 1e-12))
    return {
        "mean": mean,
        "variance_pct": variance_pct,
        "display": f"{mean:.3e} (+/- {variance_pct:.1f}%)",
    }


def _load_run_config(run: dict) -> dict:
    run_dir = Path(run["run_dir"])
    config_path = run_dir / "config.json"
    if config_path.exists():
        return unified._load_json(config_path)
    args_path = run_dir / "args.json"
    if args_path.exists():
        return unified._load_json(args_path)
    return {}


def _dataset_total_samples(dataset_dir: Path) -> int:
    dataset_path = dataset_dir / "dataset.npz"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Missing dataset file at {dataset_path}")
    with np.load(dataset_path) as data:
        return int(data["X"].shape[0])


def _optimizer_resource_metrics(dataset_dir: Path, cfg_dict: dict) -> dict[str, float | None]:
    metadata_path = dataset_dir / "metadata.json"
    if not metadata_path.exists():
        return {
            "time_s_per_batch": None,
            "time_s_per_sample": None,
            "space_mb_per_batch": None,
            "space_mb_per_sample": None,
        }

    metadata = unified._load_json(metadata_path)
    total_samples = int(metadata.get("optimizer_generation_num_points", _dataset_total_samples(dataset_dir)))
    if total_samples <= 0:
        total_samples = max(1, _dataset_total_samples(dataset_dir))

    batch_size = int(cfg_dict.get("batch_size", cfg_dict.get("batchSize", 1)))
    batch_size = max(1, batch_size)
    total_batches = int(np.ceil(total_samples / batch_size))
    total_batches = max(1, total_batches)

    wall_time = metadata.get("optimizer_generation_wall_time_sec")
    space_mb = metadata.get("optimizer_generation_space_mb")

    return {
        "time_s_per_batch": (float(wall_time) / total_batches) if wall_time is not None else None,
        "time_s_per_sample": (float(wall_time) / total_samples) if wall_time is not None else None,
        "space_mb_per_batch": (float(space_mb) / total_batches) if space_mb is not None else None,
        "space_mb_per_sample": (float(space_mb) / total_samples) if space_mb is not None else None,
    }


def _resolve_run_resource_denominators(
    *,
    framework: str,
    run: dict,
    dataset_dir: Path,
    cfg_dict: dict,
    problem_family: str,
) -> tuple[int, int]:
    run_cfg = dict(cfg_dict)
    run_cfg.update(_load_run_config(run))
    total_samples = _dataset_total_samples(dataset_dir)

    if framework == "nlpopt":
        batch_size = int(run_cfg.get("batch_size", 1))
        train_frac = float(run_cfg.get("train_frac", 0.8))
        n_train = int(train_frac * total_samples)
        n_train = (n_train // batch_size) * batch_size
        n_val = ((total_samples - int(train_frac * total_samples)) // batch_size) * batch_size
        total_batches = (n_train // batch_size) + (n_val // batch_size)
        return max(1, total_batches), max(1, n_train + n_val)

    batch_size = int(run_cfg.get("batch_size", run_cfg.get("batchSize", 1)))
    if problem_family == "nlp":
        train_frac, valid_frac, _test_frac = unified.nlp_torch._resolve_split_fracs(run_cfg)
    else:
        train_frac, valid_frac, _test_frac = unified.poly_torch._resolve_split_fracs(run_cfg)
    n_train = int(total_samples * train_frac)
    n_valid = int(total_samples * (train_frac + valid_frac)) - n_train
    total_batches = int(np.ceil(n_train / batch_size)) + int(np.ceil(n_valid / batch_size))
    return max(1, total_batches), max(1, n_train + n_valid)


def _resource_metrics_for_run(
    *,
    framework: str,
    run: dict,
    payload: dict,
    dataset_dir: Path,
    cfg_dict: dict,
    problem_family: str,
) -> dict:
    total_batches, total_samples = _resolve_run_resource_denominators(
        framework=framework,
        run=run,
        dataset_dir=dataset_dir,
        cfg_dict=cfg_dict,
        problem_family=problem_family,
    )
    time_per_batch = payload.get("avg_total_batch_time_sec")
    if time_per_batch is None and payload.get("training_wall_time_sec") is not None:
        time_per_batch = float(payload["training_wall_time_sec"]) / total_batches
    space_mb = payload.get("space_mb")
    if space_mb is None:
        run_dir = Path(run["run_dir"])
        if run_dir.exists():
            space_mb = unified._directory_size_mb(run_dir)
    return {
        "time_s_per_batch": float(time_per_batch) if time_per_batch is not None else None,
        "time_s_per_sample": (float(time_per_batch) / total_samples * total_batches) if time_per_batch is not None else None,
        "space_mb_per_batch": (float(space_mb) / total_batches) if space_mb is not None else None,
        "space_mb_per_sample": (float(space_mb) / total_samples) if space_mb is not None else None,
    }


def _optimizer_summary_for_seed(case_dir: Path, data_cfg: dict, cfg_dict: dict, seed: int) -> dict:
    seed_cfg = dict(cfg_dict)
    seed_cfg["seed"] = int(seed)
    if unified._problem_family(data_cfg) == "nlp":
        _dataset, problem = unified._prepare_nlp_problem(case_dir, data_cfg, seed_cfg)
    else:
        _dataset, problem = unified._prepare_poly_problem(case_dir, data_cfg, seed_cfg)

    X = unified.torch.cat((problem.trainX, problem.validX), dim=0)
    Y = unified.torch.cat((problem.trainY, problem.validY), dim=0)
    eq_abs, ineq_violation, _bound_violation = unified._violation_components(problem, X, Y)
    objective = problem.obj_fn(Y)
    return {
        "objective_value": float(unified.torch.mean(objective).item()),
        "max_equality": float(unified.torch.max(eq_abs).item()) if eq_abs.numel() > 0 else 0.0,
        "mean_equality": float(unified.torch.mean(eq_abs).item()) if eq_abs.numel() > 0 else 0.0,
        "max_inequality": float(unified.torch.max(ineq_violation).item()) if ineq_violation.numel() > 0 else 0.0,
        "mean_inequality": float(unified.torch.mean(ineq_violation).item()) if ineq_violation.numel() > 0 else 0.0,
        "consistency": 0.0,
        "optimality_gap": 0.0,
        "training_wall_time_sec": None,
        "space_mb": None,
    }


def _load_comparison_metric_inputs(comparison_summary: dict) -> tuple[Path, dict, dict, str, dict, dict]:
    case_dir = Path(comparison_summary["case_dir"])
    comparison_dir = Path(comparison_summary["comparison_dir"])
    data_cfg_path = comparison_dir / "data.json"
    cfg_path = comparison_dir / "config.json"
    dataset_dir = Path(comparison_summary["dataset_dir"])
    if not data_cfg_path.exists():
        data_cfg_path = comparison_dir / "data_config.json"
    if not data_cfg_path.exists():
        data_cfg_path = dataset_dir / "data.json"
    if not data_cfg_path.exists():
        data_cfg_path = dataset_dir / "data_config.json"
    if not cfg_path.exists():
        cfg_path = ROOT / "config.json"
    data_cfg = unified._load_json(data_cfg_path)
    cfg_dict = unified._load_json(cfg_path)
    framework_payloads = {}
    framework_seed_map = {}

    for framework, info in comparison_summary.get("frameworks", {}).items():
        multi_summary = unified._load_json(Path(info["summary_path"]))
        framework_seed_map[framework] = [int(seed) for seed in multi_summary.get("seeds", [])]
        run_payloads = []
        for run in multi_summary.get("runs", []):
            metrics_path = Path(run["metrics_path"])
            if not metrics_path.exists():
                raise FileNotFoundError(f"Missing metrics file at {metrics_path}")
            run_payloads.append((run, unified._load_json(metrics_path)))
        framework_payloads[framework] = run_payloads
    return case_dir, data_cfg, cfg_dict, unified._problem_family(data_cfg), dataset_dir, framework_payloads


def _build_table_payload(comparison_summary: dict) -> dict:
    case_dir, data_cfg, cfg_dict, _problem_family, _dataset_dir, framework_payloads = _load_comparison_metric_inputs(comparison_summary)
    optimizer_seeds = next(
        (
            [int(seed) for seed in unified._load_json(Path(info["summary_path"])).get("seeds", [])]
            for info in comparison_summary.get("frameworks", {}).values()
        ),
        [],
    )
    optimizer_payloads = [_optimizer_summary_for_seed(case_dir, data_cfg, cfg_dict, seed) for seed in optimizer_seeds]

    table_rows = []
    json_rows = []
    for framework_key, row_label in _ordered_table_framework_rows(comparison_summary):
        if framework_key == "optimizer":
            payloads = [(None, payload) for payload in optimizer_payloads]
        else:
            payloads = framework_payloads.get(framework_key, [])

        formatted_row = {"Model": row_label}
        json_row = {"model": row_label, "metrics": {}}
        for metric_key, column_label in _TABLE_COLUMNS:
            values = []
            for run, payload in payloads:
                metric = _metric_value(payload, metric_key, run or {})
                if metric is not None and np.isfinite(metric):
                    values.append(float(metric))
            aggregate = _aggregate_values(values)
            formatted_row[column_label] = aggregate["display"]
            json_row["metrics"][metric_key] = aggregate
        table_rows.append(formatted_row)
        json_rows.append(json_row)

    return {
        "columns": ["Model"] + [label for _, label in _TABLE_COLUMNS],
        "rows": table_rows,
        "rows_json": json_rows,
    }


def _build_resource_table_payload(comparison_summary: dict) -> dict:
    _case_dir, _data_cfg, cfg_dict, problem_family, dataset_dir, framework_payloads = _load_comparison_metric_inputs(comparison_summary)
    table_rows = []
    json_rows = []

    for framework_key, row_label in _ordered_table_framework_rows(comparison_summary):
        formatted_row = {"Model": row_label}
        json_row = {"model": row_label, "metrics": {}}
        if framework_key == "optimizer":
            resource_runs = [_optimizer_resource_metrics(dataset_dir, cfg_dict)]
        else:
            payloads = framework_payloads.get(framework_key, [])
            resource_runs = []
            for run, payload in payloads:
                resource_runs.append(
                    _resource_metrics_for_run(
                        framework=framework_key,
                        run=run,
                        payload=payload,
                        dataset_dir=dataset_dir,
                        cfg_dict=cfg_dict,
                        problem_family=problem_family,
                    )
                )

        for metric_key, column_label in _RESOURCE_COLUMNS:
            values = []
            for resource_payload in resource_runs:
                metric = resource_payload.get(metric_key)
                if metric is not None and np.isfinite(metric):
                    values.append(float(metric))
            aggregate = _aggregate_values(values)
            formatted_row[column_label] = aggregate["display"]
            json_row["metrics"][metric_key] = aggregate
        table_rows.append(formatted_row)
        json_rows.append(json_row)

    return {
        "columns": ["Model"] + [label for _, label in _RESOURCE_COLUMNS],
        "rows": table_rows,
        "rows_json": json_rows,
    }


def _write_table_outputs(comparison_dir: Path, table_payload: dict, *, stem: str) -> dict:
    csv_path = comparison_dir / f"{stem}.csv"
    md_path = comparison_dir / f"{stem}.md"
    json_path = comparison_dir / f"{stem}.json"

    with open(csv_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=table_payload["columns"])
        writer.writeheader()
        writer.writerows(table_payload["rows"])

    if md_path.exists():
        md_path.unlink()
    if json_path.exists():
        json_path.unlink()
    return {"csv_path": str(csv_path)}


def plot_root_comparison(dataset_dir_arg: str | None = None) -> int:
    dataset_dir, comparison_dir = _resolve_dataset_or_comparison_dir(dataset_dir_arg)
    summary_path = comparison_dir / "comparison_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Expected comparison summary at {summary_path}. Run the corresponding comparison entrypoint or main.py --action compare first."
        )

    comparison_summary = unified._load_json(summary_path)
    epochs, model_runs = _load_framework_histories(comparison_summary)
    output_path = comparison_dir / "compare_models_shadow.png"
    save_comparison_shadow_plot(output_path, epochs=epochs, model_runs=model_runs)

    table_payload = _build_table_payload(comparison_summary)
    table_paths = _write_table_outputs(comparison_dir, table_payload, stem="comparison_table")
    resource_payload = _build_resource_table_payload(comparison_summary)
    resource_paths = _write_table_outputs(comparison_dir, resource_payload, stem="comparison_resources")

    unified._write_json(
        comparison_dir / "plot_comparison_summary.json",
        {
            "comparison_summary_path": str(summary_path),
            "dataset_dir": str(dataset_dir),
            "comparison_dir": str(comparison_dir),
            "plot_path": str(output_path),
            "models": list(model_runs.keys()),
            "epochs": epochs,
            "comparison_table_csv_path": table_paths["csv_path"],
            "comparison_resources_csv_path": resource_paths["csv_path"],
        },
    )
    print(f"[comparison] Saved plot: {output_path}")
    print(f"[comparison] Saved table: {table_paths['csv_path']}")
    print(f"[comparison] Saved resources: {resource_paths['csv_path']}")
    return 0


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the comparison shadow plot and aggregated metrics table."
    )
    parser.add_argument(
        "dataset_or_comparison_dir",
        nargs="?",
        help="Optional dataset directory or comparison directory.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args(sys.argv[1:])
    raise SystemExit(plot_root_comparison(args.dataset_or_comparison_dir))
