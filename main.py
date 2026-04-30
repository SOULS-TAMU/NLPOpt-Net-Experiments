#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from scripts.misc.cli_overrides import set_cli_overrides


ROOT = Path(__file__).resolve().parent

_TYPE_TO_DIR = {
    "qp": ROOT / "case" / "qp",
    "qcqp": ROOT / "case" / "qcqp",
    "nlp": ROOT / "case" / "nlp",
    "nonconvex": ROOT / "case" / "nonconvx",
    "nonconvx": ROOT / "case" / "nonconvx",
    "custom": ROOT / "case" / "custom",
    "general": ROOT / "case" / "general",
}

_ACTION_ALIASES = {
    "dc3_compare": "compare_dc3",
}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Root dispatcher for the standalone case folders. "
            "It loads data/config/proj from the selected case/* directory "
            "and saves outputs in that directory's problem_data tree."
        )
    )
    parser.add_argument(
        "--type",
        required=True,
        choices=("qp", "qcqp", "nlp", "nonconvex", "nonconvx", "custom", "general"),
        help="Problem family to run.",
    )
    parser.add_argument(
        "--action",
        default="run",
        choices=("run", "compare", "compare_dc3", "dc3_compare", "plot-comparison", "plot-time"),
        help="Which shared entrypoint to invoke. Default: run.",
    )
    parser.add_argument(
        "--path",
        default=None,
        help=(
            "Optional dataset/comparison path for plot actions. "
            "If omitted, the selected case folder will use its local config to resolve the default path."
        ),
    )
    parser.add_argument("--p", type=int, default=None, help="Optional override for p or n_x in data.json.")
    parser.add_argument("--n", type=int, default=None, help="Optional override for n or n_y in data.json.")
    parser.add_argument("--me", type=int, default=None, help="Optional override for me or n_eq in data.json.")
    parser.add_argument("--mi", type=int, default=None, help="Optional override for mi or n_ineq in data.json.")
    parser.add_argument(
        "--train_frac",
        type=float,
        default=None,
        help="Optional override for train_frac in config.json.",
    )
    return parser.parse_args(argv)


def _resolve_dispatch(problem_type: str, action: str):
    action = _ACTION_ALIASES.get(action, action)
    if problem_type in {"qp", "qcqp", "nlp"}:
        if action == "run":
            from scripts.testcase import standard as entrypoints
            return entrypoints.run_case
        from scripts.comparison import standard as entrypoints
        return {
            "compare": entrypoints.run_comparison,
            "compare_dc3": entrypoints.run_comparison_dc3,
            "plot-comparison": entrypoints.plot_comparison,
            "plot-time": entrypoints.plot_time_comparison,
        }[action]

    if problem_type in {"nonconvex", "nonconvx"}:
        if action == "run":
            from scripts.testcase import nonconvx as entrypoints
            return entrypoints.run_case
        from scripts.comparison import nonconvx as entrypoints
        return {
            "compare": entrypoints.run_comparison,
            "compare_dc3": entrypoints.run_comparison_dc3,
            "plot-comparison": entrypoints.plot_comparison,
            "plot-time": entrypoints.plot_time_comparison,
        }[action]

    if problem_type == "custom":
        if action == "compare_dc3":
            raise ValueError("The compare_dc3 action is only supported for qp, qcqp, nlp, and nonconvex cases.")
        if action == "run":
            from scripts.testcase import custom_run as custom_runner
            return custom_runner.run_case
        from scripts.comparison import custom as custom_entrypoints
        return {
            "compare": custom_entrypoints.run_comparison,
            "plot-comparison": custom_entrypoints.plot_comparison,
            "plot-time": custom_entrypoints.plot_time_comparison,
        }[action]

    if problem_type == "general":
        if action == "compare_dc3":
            raise ValueError("The compare_dc3 action is only supported for qp, qcqp, nlp, and nonconvex cases.")
        if action == "run":
            from scripts.testcase import general_run as general_runner
            return general_runner.run_case
        from scripts.comparison import general as general_entrypoints
        return {
            "compare": general_entrypoints.run_comparison,
            "plot-comparison": general_entrypoints.plot_comparison,
            "plot-time": general_entrypoints.plot_time_comparison,
        }[action]

    raise ValueError(f"Unsupported type: {problem_type}")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    canonical_action = _ACTION_ALIASES.get(args.action, args.action)
    set_cli_overrides(
        p=args.p,
        n=args.n,
        me=args.me,
        mi=args.mi,
        train_frac=args.train_frac,
    )
    case_dir = _TYPE_TO_DIR[args.type]
    if not case_dir.exists():
        raise FileNotFoundError(f"Expected case directory at {case_dir}")

    dispatch = _resolve_dispatch(args.type, canonical_action)

    print("")
    print("=" * 80)
    print(f"Root dispatcher | type={args.type} action={canonical_action}")
    print(f"Using case directory: {case_dir}")
    if args.path is not None:
        print(f"Forwarded path argument: {args.path}")
    print("=" * 80)

    return int(dispatch(case_dir, args.path) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
