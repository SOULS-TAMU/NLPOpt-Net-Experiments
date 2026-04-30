#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.plot_utils import plotting as shared_plotting  # noqa: E402
from scripts.testcase import unified_runner as unified  # noqa: E402

_COMPONENTS = (
    ("forward", "time_backbone_percent", "Backbone"),
    ("projection", "time_projection_percent", "Projection"),
    ("backward", "time_backward_percent", "Backward"),
    ("optimizer", "time_optimizer_percent", "Adam"),
)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a component-time box plot from a saved comparison directory."
    )
    parser.add_argument(
        "comparison_dir",
        help=(
            "Comparison root directory such as "
            "test/problem_data/qp/.../comparison, "
            "test/problem_data/qp/.../comparison_dc3, "
            "or a nested nlpopt_vs_dc3 directory."
        ),
    )
    return parser.parse_args(argv)


def _resolve_comparison_paths(path_arg: str) -> tuple[Path, Path]:
    input_path = Path(path_arg).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Path does not exist: {input_path}")

    summary_in_place = input_path / "comparison_summary.json"
    if summary_in_place.exists():
        output_dir = input_path.parent if input_path.name == "nlpopt_vs_dc3" else input_path
        return input_path, output_dir

    summary_in_child = input_path / "nlpopt_vs_dc3" / "comparison_summary.json"
    if summary_in_child.exists():
        return input_path / "nlpopt_vs_dc3", input_path

    summary_dc3_root = input_path / "comparison_dc3" / "comparison_summary.json"
    if summary_dc3_root.exists():
        return input_path / "comparison_dc3", input_path / "comparison_dc3"

    dc3_root = input_path / "comparison_dc3"
    if dc3_root.exists():
        dc3_candidates = sorted(
            child for child in dc3_root.iterdir()
            if child.is_dir() and (child / "comparison_summary.json").exists()
        )
        if len(dc3_candidates) == 1:
            return dc3_candidates[0], dc3_candidates[0]

    summary_from_dataset = input_path / "comparison" / "nlpopt_vs_dc3" / "comparison_summary.json"
    if summary_from_dataset.exists():
        return input_path / "comparison" / "nlpopt_vs_dc3", input_path / "comparison"

    raise FileNotFoundError(
        "Could not find comparison_summary.json from the provided path. "
        "Pass a comparison directory such as .../comparison, .../comparison_dc3, or .../comparison/nlpopt_vs_dc3."
    )


def _epoch_count_from_history(history_path: Path) -> int:
    history = unified._load_json(history_path)
    epochs = history.get("epochs", [])
    return max(1, len(epochs))


def _avg_total_epoch_time(metrics_payload: dict, history_path: Path) -> float:
    if "avg_total_epoch_time_sec" in metrics_payload:
        return max(float(metrics_payload["avg_total_epoch_time_sec"]), 1e-12)

    if "avg_train_epoch_time_sec" in metrics_payload and "avg_val_epoch_time_sec" in metrics_payload:
        return max(
            float(metrics_payload["avg_train_epoch_time_sec"]) + float(metrics_payload["avg_val_epoch_time_sec"]),
            1e-12,
        )

    if "training_wall_time_sec" in metrics_payload:
        epochs_recorded = metrics_payload.get("timing_epochs_recorded", metrics_payload.get("epochs_recorded"))
        if epochs_recorded is None:
            epochs_recorded = _epoch_count_from_history(history_path)
        return max(float(metrics_payload["training_wall_time_sec"]) / int(epochs_recorded), 1e-12)

    raise KeyError("Unable to recover average epoch time from saved metrics payload.")


def _component_time_per_epoch_seconds(metrics_payload: dict, history_path: Path, percent_key: str, raw_key: str | None = None) -> float:
    epochs_recorded = metrics_payload.get("timing_epochs_recorded", metrics_payload.get("epochs_recorded"))
    if raw_key is not None and raw_key in metrics_payload and epochs_recorded is not None:
        return max(float(metrics_payload[raw_key]) / max(1, int(epochs_recorded)), 1e-12)

    avg_epoch_time = _avg_total_epoch_time(metrics_payload, history_path)
    percent = float(metrics_payload.get(percent_key, 0.0))
    return max(avg_epoch_time * percent / 100.0, 1e-12)


def _display_label_for_framework(framework: str) -> str:
    normalized = str(framework).strip().lower()
    if normalized == "nlpopt":
        return "NLPOpt-Net"
    if normalized == "dc3":
        return "DC3"
    if normalized == "baseline_nn":
        return "Baseline NN"
    if normalized == "baseline_eq_nn":
        return "EqNN"
    return unified._framework_label(framework)


def _collect_time_payloads(comparison_summary_path: Path) -> tuple[list[str], dict[str, dict[str, list[float]]]]:
    comparison_summary = unified._load_json(comparison_summary_path)
    frameworks = list(comparison_summary.get("frameworks", {}).keys())
    if not frameworks:
        raise ValueError(f"No frameworks found in {comparison_summary_path}")

    component_times = {framework: {name: [] for name, _, _ in _COMPONENTS} for framework in frameworks}

    for framework in frameworks:
        framework_info = comparison_summary["frameworks"].get(framework)
        if framework_info is None:
            raise ValueError(f"Missing framework entry '{framework}' in {comparison_summary_path}")

        multi_seed_summary = unified._load_json(Path(framework_info["summary_path"]))
        for run in multi_seed_summary.get("runs", []):
            history_path = Path(run["history_path"])
            metrics_path = Path(run["metrics_path"])
            if not history_path.exists():
                raise FileNotFoundError(f"Missing history file at {history_path}")
            if not metrics_path.exists():
                raise FileNotFoundError(f"Missing metrics file at {metrics_path}")

            metrics_payload = unified._load_json(metrics_path)
            component_times[framework]["forward"].append(
                _component_time_per_epoch_seconds(metrics_payload, history_path, "time_backbone_percent", raw_key="backbone_total_sec")
            )
            component_times[framework]["projection"].append(
                _component_time_per_epoch_seconds(metrics_payload, history_path, "time_projection_percent", raw_key="projection_total_sec")
            )
            component_times[framework]["backward"].append(
                _component_time_per_epoch_seconds(metrics_payload, history_path, "time_backward_percent", raw_key="backward_total_sec")
            )
            component_times[framework]["optimizer"].append(
                _component_time_per_epoch_seconds(metrics_payload, history_path, "time_optimizer_percent", raw_key="optimizer_total_sec")
            )

    return frameworks, component_times


def _plot_component_boxplot(
    output_path: Path,
    frameworks: list[str],
    component_times: dict[str, dict[str, list[float]]],
) -> Path:
    plt.rcParams.update({
        "font.size": shared_plotting._PLOT_FONT_SIZE,
        "axes.linewidth": 1.5,
        "xtick.major.size": 6,
        "xtick.major.width": 1.5,
        "ytick.major.size": 6,
        "ytick.major.width": 1.5,
        "legend.frameon": False,
    })

    fig, ax = plt.subplots(1, 1, figsize=(8.4, 6.2))
    fig.patch.set_facecolor("white")

    centers = [1.0, 2.1, 3.2, 4.3]
    framework_count = len(frameworks)
    width = min(0.22, 0.72 / max(1, framework_count))
    if framework_count == 1:
        shifts = [0.0]
    else:
        span = width * (framework_count - 1) * 1.35
        shifts = list(np.linspace(-span / 2.0, span / 2.0, framework_count))

    framework_colors = {
        framework: shared_plotting._model_color(_display_label_for_framework(framework))
        for framework in frameworks
    }

    for framework, shift in zip(frameworks, shifts):
        positions = [center + shift for center in centers]
        series = [component_times[framework][component_name] for component_name, _, _ in _COMPONENTS]
        color = framework_colors[framework]
        boxplot = ax.boxplot(
            series,
            positions=positions,
            widths=width,
            patch_artist=True,
            showfliers=False,
            whis=(5, 95),
        )
        for patch in boxplot["boxes"]:
            patch.set(facecolor=color, edgecolor=color, alpha=0.28, linewidth=1.8)
        for whisker in boxplot["whiskers"]:
            whisker.set(color=color, linewidth=1.6)
        for cap in boxplot["caps"]:
            cap.set(color=color, linewidth=1.6)
        for median in boxplot["medians"]:
            median.set(color=color, linewidth=2.0)

    ax.set_xticks(centers)
    ax.set_xticklabels([label for _, _, label in _COMPONENTS], fontsize=shared_plotting._LABEL_FONT_SIZE)
    ax.set_ylabel("Time per Epoch (s)", fontsize=shared_plotting._LABEL_FONT_SIZE)
    ax.grid(True, axis="y", linestyle="--", alpha=shared_plotting._GRID_ALPHA)
    ax.set_xlim(0.55, 4.75)

    legend_handles = [
        Patch(
            facecolor=framework_colors[framework],
            edgecolor=framework_colors[framework],
            alpha=0.28,
            label=_display_label_for_framework(framework),
        )
        for framework in frameworks
    ]
    fig.legend(
        legend_handles,
        [_display_label_for_framework(framework) for framework in frameworks],
        loc="lower center",
        ncol=min(4, max(1, len(frameworks))),
        frameon=False,
        bbox_to_anchor=(0.5, -0.03),
        fontsize=shared_plotting._LEGEND_FONT_SIZE,
    )

    fig.subplots_adjust(bottom=0.18)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(output_path, bbox_inches="tight", dpi=600)
    plt.close(fig)
    print(f"[plot] Saved: {output_path}")
    return output_path


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    comparison_summary_dir, output_dir = _resolve_comparison_paths(args.comparison_dir)
    summary_path = comparison_summary_dir / "comparison_summary.json"
    frameworks, component_times = _collect_time_payloads(summary_path)

    output_path = output_dir / "compare_time_boxplot.png"
    _plot_component_boxplot(output_path, frameworks, component_times)

    unified._write_json(
        output_dir / "compare_time_boxplot_summary.json",
        {
            "comparison_summary_path": str(summary_path),
            "plot_path": str(output_path),
            "components": [label for _, _, label in _COMPONENTS],
            "models": [_display_label_for_framework(framework) for framework in frameworks],
            "time_metric": "per_epoch_seconds",
        },
    )
    print(f"[comparison] Saved time plot: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
