#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import sys
from pathlib import Path

import numpy as np

from scripts.testcase import custom_run as local_runner
from scripts.plot_utils.plotting import save_comparison_shadow_plot
from scripts.comparison import plot_comparison as base_plot
from scripts.comparison import plot_time_comparison as time_plot
from scripts.testcase import unified_runner as unified

_FRAMEWORK = "nlpopt"
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


def _default_dataset_dir() -> Path:
    return local_runner.default_dataset_dir()


def _resolve_dataset_or_comparison_dir(path_arg: str | None) -> tuple[Path, Path]:
    if path_arg is None:
        dataset_dir = _default_dataset_dir()
        return dataset_dir, dataset_dir / "comparison"

    supplied = Path(path_arg).expanduser().resolve()
    if not supplied.exists():
        raise FileNotFoundError(f"Provided path does not exist: {supplied}")

    if supplied.is_dir() and (supplied / "comparison_summary.json").exists():
        return supplied.parent, supplied

    if supplied.is_dir() and (supplied / "comparison" / "comparison_summary.json").exists():
        return supplied, supplied / "comparison"

    if supplied.is_dir() and (supplied / "nlpopt_vs_dc3" / "comparison_summary.json").exists():
        return supplied.parent, supplied / "nlpopt_vs_dc3"

    return supplied, supplied / "comparison"


def _write_csv_table(path: Path, *, columns: list[str], rows: list[dict[str, str]]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _comparison_metric_rows(comparison_summary: dict) -> list[dict[str, str]]:
    rows = []
    for framework, info in comparison_summary.get("frameworks", {}).items():
        multi_summary = unified._load_json(Path(info["summary_path"]))
        metrics_payloads = []
        for run in multi_summary.get("runs", []):
            metrics_path = Path(run["metrics_path"])
            if metrics_path.exists():
                metrics_payloads.append(unified._load_json(metrics_path))

        row = {"Model": info.get("label", unified._framework_label(framework))}
        for metric_key, column_label in _TABLE_COLUMNS:
            values = []
            for payload in metrics_payloads:
                metric = base_plot._metric_value(payload, metric_key, {})
                if metric is not None and np.isfinite(metric):
                    values.append(float(metric))
            row[column_label] = base_plot._aggregate_values(values)["display"]
        rows.append(row)
    return rows


def _comparison_resource_rows(comparison_summary: dict, dataset_dir: Path, cfg_dict: dict) -> list[dict[str, str]]:
    rows = []

    optimizer_payload = base_plot._optimizer_resource_metrics(dataset_dir, cfg_dict)
    optimizer_row = {"Model": "Optimizer"}
    for metric_key, column_label in _RESOURCE_COLUMNS:
        metric = optimizer_payload.get(metric_key)
        values = [float(metric)] if metric is not None and np.isfinite(metric) else []
        optimizer_row[column_label] = base_plot._aggregate_values(values)["display"]
    rows.append(optimizer_row)

    for framework, info in comparison_summary.get("frameworks", {}).items():
        multi_summary = unified._load_json(Path(info["summary_path"]))
        resource_payloads = []
        for run in multi_summary.get("runs", []):
            metrics_path = Path(run["metrics_path"])
            if not metrics_path.exists():
                continue
            payload = unified._load_json(metrics_path)
            resource_payloads.append(
                base_plot._resource_metrics_for_run(
                    framework=framework,
                    run=run,
                    payload=payload,
                    dataset_dir=dataset_dir,
                    cfg_dict=cfg_dict,
                    problem_family="nlp",
                )
            )

        row = {"Model": info.get("label", unified._framework_label(framework))}
        for metric_key, column_label in _RESOURCE_COLUMNS:
            values = []
            for payload in resource_payloads:
                metric = payload.get(metric_key)
                if metric is not None and np.isfinite(metric):
                    values.append(float(metric))
            row[column_label] = base_plot._aggregate_values(values)["display"]
        rows.append(row)
    return rows


def run_comparison(_case_dir: Path | None = None, _path_arg: str | None = None) -> int:
    case_dir = local_runner._case_workspace()
    data_cfg, cfg_dict, proj_cfg, problem_cfg = local_runner._load_local_configs()
    data_cfg = local_runner._normalize_local_data_cfg(data_cfg, problem_cfg)
    problem_cfg = local_runner._normalize_problem_cfg(problem_cfg)
    dataset_dir = local_runner.dataset_dir(case_dir, data_cfg, problem_cfg)
    comparison_dir = unified._comparison_dir(dataset_dir)
    comparison_dir.mkdir(parents=True, exist_ok=True)

    base_seed = int(cfg_dict.get("seed", 42))
    num_seeds = int(cfg_dict.get("num_seeds", 10))
    if num_seeds <= 0:
        raise ValueError("num_seeds must be positive for comparison runs.")

    run_cfg = copy.deepcopy(cfg_dict)
    run_cfg["model"] = _FRAMEWORK
    run_cfg["run_multiple_seed"] = True

    print("")
    print("=" * 80)
    print("Comparison runner | framework=NLPOpt")
    print(f"Dataset target: {dataset_dir}")
    print(f"Seeds: {base_seed}..{base_seed + num_seeds - 1} ({num_seeds} runs)")
    print("=" * 80)

    local_runner._run_multi_seed(case_dir, data_cfg, run_cfg, proj_cfg, problem_cfg)

    multi_dir = unified._framework_multi_dir(dataset_dir, _FRAMEWORK)
    multi_summary_path = multi_dir / "multi_seed_summary.json"
    if not multi_summary_path.exists():
        raise FileNotFoundError(f"Expected multi-seed summary at {multi_summary_path}")
    multi_summary = unified._load_json(multi_summary_path)

    framework_manifests = {
        _FRAMEWORK: {
            "label": "NLPOpt",
            "summary_path": str(multi_summary_path),
            "multi_dir": str(multi_dir),
            "shadow_plot_path": multi_summary.get("shadow_plot_path"),
            "seeds": multi_summary.get("seeds", []),
            "num_runs": len(multi_summary.get("runs", [])),
        }
    }

    unified._write_run_configs(comparison_dir, data_cfg, cfg_dict, proj_cfg)
    local_runner._write_json(comparison_dir / "problem.json", problem_cfg)
    summary_path = comparison_dir / "comparison_summary.json"
    local_runner._write_json(
        summary_path,
        {
            "comparison_name": "multi_model_comparison",
            "case_dir": str(case_dir),
            "dataset_dir": str(dataset_dir),
            "comparison_dir": str(comparison_dir),
            "problem_type": str(data_cfg["type"]).lower(),
            "base_seed": base_seed,
            "num_seeds": num_seeds,
            "frameworks": framework_manifests,
        },
    )

    plot_comparison(None, str(dataset_dir))
    metadata_path = unified._append_family_metadata(
        dataset_dir,
        mode="comparison",
        output_dir=comparison_dir,
        data_cfg=data_cfg,
        cfg_dict=cfg_dict,
        proj_cfg=proj_cfg,
        frameworks=(_FRAMEWORK,),
        seeds=[base_seed + idx for idx in range(num_seeds)],
        extra={"summary_path": str(summary_path), "problem": problem_cfg},
    )

    print("")
    print(f"[comparison] Saved summary: {summary_path}")
    print(f"[comparison] Updated family metadata: {metadata_path}")
    return 0


def plot_root_comparison(dataset_dir_arg: str | None = None) -> int:
    dataset_dir, comparison_dir = _resolve_dataset_or_comparison_dir(dataset_dir_arg)
    summary_path = comparison_dir / "comparison_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Expected comparison summary at {summary_path}")

    comparison_summary = unified._load_json(summary_path)
    epochs, model_runs = base_plot._load_framework_histories(comparison_summary)
    output_path = comparison_dir / "compare_models_shadow.png"
    save_comparison_shadow_plot(output_path, epochs=epochs, model_runs=model_runs)

    cfg_path = comparison_dir / "config.json"
    if not cfg_path.exists():
        cfg_path = Path(__file__).resolve().parents[2] / "case" / "custom" / "config.json"
    cfg_dict = unified._load_json(cfg_path)

    table_columns = ["Model"] + [label for _, label in _TABLE_COLUMNS]
    resource_columns = ["Model"] + [label for _, label in _RESOURCE_COLUMNS]
    table_rows = _comparison_metric_rows(comparison_summary)
    resource_rows = _comparison_resource_rows(comparison_summary, dataset_dir, cfg_dict)

    table_path = comparison_dir / "comparison_table.csv"
    resource_path = comparison_dir / "comparison_resources.csv"
    _write_csv_table(table_path, columns=table_columns, rows=table_rows)
    _write_csv_table(resource_path, columns=resource_columns, rows=resource_rows)

    unified._write_json(
        comparison_dir / "plot_comparison_summary.json",
        {
            "comparison_summary_path": str(summary_path),
            "dataset_dir": str(dataset_dir),
            "comparison_dir": str(comparison_dir),
            "plot_path": str(output_path),
            "models": list(model_runs.keys()),
            "epochs": epochs,
            "comparison_table_csv_path": str(table_path),
            "comparison_resources_csv_path": str(resource_path),
        },
    )
    print(f"[comparison] Saved plot: {output_path}")
    print(f"[comparison] Saved table: {table_path}")
    print(f"[comparison] Saved resources: {resource_path}")
    return 0


def plot_comparison(_case_dir: Path | None = None, dataset_dir_arg: str | None = None) -> int:
    return plot_root_comparison(dataset_dir_arg)


def plot_time_comparison(_case_dir: Path | None = None, comparison_dir_arg: str | None = None) -> int:
    target = comparison_dir_arg if comparison_dir_arg is not None else str(_default_dataset_dir() / "comparison")
    return time_plot.main([target])


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the custom comparison shadow plot and summary CSVs.")
    parser.add_argument("dataset_or_comparison_dir", nargs="?", help="Optional dataset or comparison directory.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    parsed = _parse_args(args)
    return plot_root_comparison(parsed.dataset_or_comparison_dir)
