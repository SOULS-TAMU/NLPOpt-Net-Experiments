#!/usr/bin/env python3
from __future__ import annotations

import copy
from pathlib import Path

from scripts.comparison import plot_comparison as comparison_plotter
from scripts.comparison import plot_time_comparison as time_plotter
from scripts.testcase import nonconvx as nonconvx_case
from scripts.testcase import nonconvx_run as base_runner

_FRAMEWORKS = ("nlpopt", "dc3", "baseline_nn", "baseline_eq_nn")
_DC3_FRAMEWORKS = ("nlpopt", "dc3")


def _train_frac_dir_name(cfg_dict: dict) -> str:
    raw_value = float(cfg_dict.get("train_frac", 0.8))
    text = f"{raw_value:.6f}".rstrip("0").rstrip(".")
    if "." not in text:
        text = f"{text}.0"
    return f"train_frac_{text.replace('.', 'p')}"

def _run_framework_comparison(
    case_dir: Path,
    *,
    frameworks: tuple[str, ...],
    comparison_name: str,
    comparison_dir_name: str | None,
    metadata_mode: str,
) -> int:
    raw_data_cfg, cfg_dict, proj_cfg = nonconvx_case.load_local_configs(case_dir)
    data_cfg = nonconvx_case.normalize_local_data_cfg(raw_data_cfg)

    dataset_root = base_runner.dataset_dir(case_dir, data_cfg)
    if comparison_dir_name == "comparison_dc3":
        comparison_dir = dataset_root / comparison_dir_name / _train_frac_dir_name(cfg_dict)
    else:
        comparison_dir = (
            dataset_root / comparison_dir_name
            if comparison_dir_name
            else base_runner.unified._comparison_dir(dataset_root)
        )
    comparison_dir.mkdir(parents=True, exist_ok=True)

    base_seed = int(cfg_dict.get("seed", 42))
    num_seeds = int(cfg_dict.get("num_seeds", 10))
    if num_seeds <= 0:
        raise ValueError("num_seeds must be positive for comparison runs.")

    framework_manifests = {}
    for framework in frameworks:
        run_cfg = copy.deepcopy(cfg_dict)
        run_cfg["model"] = framework
        run_cfg["run_multiple_seed"] = True

        print("")
        print("=" * 80)
        print(f"Comparison runner | framework={base_runner.unified._framework_label(framework)}")
        print(
            f"NONCONVEX  p={int(raw_data_cfg['p'])} n={int(raw_data_cfg['n'])} "
            f"me={int(raw_data_cfg['me'])} mi={int(raw_data_cfg['mi'])}"
        )
        print(f"Dataset target: {dataset_root}")
        print(f"Seeds: {base_seed}..{base_seed + num_seeds - 1} ({num_seeds} runs)")
        print("=" * 80)

        framework_multi_dir = comparison_dir / framework / "multi" if comparison_dir_name == "comparison_dc3" else None
        base_runner._run_multi_seed(case_dir, data_cfg, run_cfg, proj_cfg, multi_dir_override=framework_multi_dir)

        multi_dir = Path(framework_multi_dir) if framework_multi_dir is not None else base_runner.unified._framework_multi_dir(dataset_root, framework)
        summary_path = multi_dir / "multi_seed_summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"Expected multi-seed summary at {summary_path}")
        summary_payload = base_runner._load_json(summary_path)
        framework_manifests[framework] = {
            "label": base_runner.unified._framework_label(framework),
            "summary_path": str(summary_path),
            "multi_dir": str(multi_dir),
            "shadow_plot_path": summary_payload.get("shadow_plot_path"),
            "seeds": summary_payload.get("seeds", []),
            "num_runs": len(summary_payload.get("runs", [])),
        }

    base_runner.unified._write_run_configs(comparison_dir, data_cfg, cfg_dict, proj_cfg)
    summary_path = comparison_dir / "comparison_summary.json"
    base_runner.unified._write_json(
        summary_path,
        {
            "comparison_name": comparison_name,
            "case_dir": str(case_dir),
            "dataset_dir": str(dataset_root),
            "comparison_dir": str(comparison_dir),
            "problem_type": str(data_cfg["type"]).lower(),
            "base_seed": base_seed,
            "num_seeds": num_seeds,
            "frameworks": framework_manifests,
        },
    )

    plot_comparison(case_dir, str(comparison_dir))
    metadata_path = base_runner.unified._append_family_metadata(
        dataset_root,
        mode=metadata_mode,
        output_dir=comparison_dir,
        data_cfg=data_cfg,
        cfg_dict=cfg_dict,
        proj_cfg=proj_cfg,
        frameworks=frameworks,
        seeds=[base_seed + idx for idx in range(num_seeds)],
        extra={"summary_path": str(summary_path)},
    )
    print("")
    print(f"[comparison] Saved summary: {summary_path}")
    print(f"[comparison] Updated family metadata: {metadata_path}")
    return 0


def run_comparison(case_dir: Path, _path_arg: str | None = None) -> int:
    return _run_framework_comparison(
        case_dir,
        frameworks=_FRAMEWORKS,
        comparison_name="multi_model_comparison",
        comparison_dir_name=None,
        metadata_mode="comparison",
    )


def run_comparison_dc3(case_dir: Path, _path_arg: str | None = None) -> int:
    return _run_framework_comparison(
        case_dir,
        frameworks=_DC3_FRAMEWORKS,
        comparison_name="nlpopt_vs_dc3",
        comparison_dir_name="comparison_dc3",
        metadata_mode="comparison_dc3",
    )


def plot_comparison(case_dir: Path, dataset_dir_arg: str | None = None) -> int:
    target = dataset_dir_arg if dataset_dir_arg is not None else str(nonconvx_case.default_dataset_dir(case_dir))
    with base_runner._patch_unified_for_nonconvex():
        return comparison_plotter.plot_root_comparison(target)


def plot_time_comparison(case_dir: Path, comparison_dir_arg: str | None = None) -> int:
    target = comparison_dir_arg if comparison_dir_arg is not None else str(nonconvx_case.default_comparison_dir(case_dir))
    return time_plotter.main([target])
