#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Optional, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.plot_utils.plotting import save_shadow_objective_violation_plot
from scripts.misc.poly_dataset_cache import dataset_dir as resolve_dataset_dir
from scripts.testcase.poly_run import run_case


def _load_json(path: Path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _config_hash(cfg_dict: dict, proj_cfg: dict) -> str:
    payload = json.dumps({"config": cfg_dict, "proj": proj_cfg}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]


def _parse_seed_text(text: str) -> list[int]:
    seeds = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if chunk:
            seeds.append(int(chunk))
    if not seeds:
        raise ValueError("Expected at least one seed.")
    return seeds


def _resolve_seeds(base_seed: int, num_runs: int, seed_step: int, explicit: Optional[str]) -> list[int]:
    if explicit is not None:
        return _parse_seed_text(explicit)
    return [base_seed + idx * seed_step for idx in range(num_runs)]


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        shutil.copy2(src, dst)


def run_multi_seed_case(
    case_dir: Path,
    *,
    seeds: Sequence[int],
) -> int:
    case_dir = case_dir.resolve()
    data_cfg = _load_json(case_dir / "data.json")
    cfg_dict = _load_json(case_dir / "config.json")
    proj_cfg = _load_json(case_dir / "proj.json")

    dataset_dir = resolve_dataset_dir(case_dir, data_cfg)
    multi_seed_dir = dataset_dir / "multi_seed_runs"
    multi_seed_dir.mkdir(parents=True, exist_ok=True)

    run_manifests = []
    history_payloads = []

    for seed in seeds:
        run_cfg = copy.deepcopy(cfg_dict)
        run_cfg["seed"] = int(seed)
        run_hash = _config_hash(run_cfg, proj_cfg)

        print("")
        print(f"[multi-seed] Running {case_dir.name} with config.seed={seed}")
        run_case(
            case_dir,
            data_cfg_override=data_cfg,
            cfg_dict_override=run_cfg,
            proj_cfg_override=proj_cfg,
        )

        params_path = dataset_dir / f"trained_params_{run_hash}.npz"
        metrics_path = dataset_dir / f"run_metrics_{run_hash}.json"
        history_path = dataset_dir / f"run_history_{run_hash}.json"
        single_plot_path = dataset_dir / "compare_metrics.png"

        seed_dir = multi_seed_dir / f"seed_{int(seed):04d}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        seed_plot_copy = seed_dir / "compare_metrics.png"
        seed_metrics_copy = seed_dir / metrics_path.name
        seed_history_copy = seed_dir / history_path.name
        _copy_if_exists(single_plot_path, seed_plot_copy)
        _copy_if_exists(metrics_path, seed_metrics_copy)
        _copy_if_exists(history_path, seed_history_copy)

        manifest = {
            "seed": int(seed),
            "run_hash": run_hash,
            "params_path": str(params_path),
            "metrics_path": str(metrics_path),
            "history_path": str(history_path),
            "plot_copy_path": str(seed_plot_copy),
        }
        with open(seed_dir / "manifest.json", "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2, sort_keys=True)
        run_manifests.append(manifest)

        with open(history_path, "r", encoding="utf-8") as fh:
            history_payloads.append(json.load(fh))

    if not history_payloads:
        raise ValueError("No histories were collected from multi-seed runs.")

    epochs = history_payloads[0]["epochs"]
    for payload in history_payloads[1:]:
        if payload["epochs"] != epochs:
            raise ValueError("All runs must have the same epoch checkpoints for shadow plotting.")

    shadow_plot_path = multi_seed_dir / "compare_metrics_shadow.png"
    save_shadow_objective_violation_plot(
        shadow_plot_path,
        epochs=epochs,
        train_gap_pct_runs=[payload["train_worst_relative_gap_pct"] for payload in history_payloads],
        val_gap_pct_runs=[payload["val_worst_relative_gap_pct"] for payload in history_payloads],
        train_violation_runs=[payload["train_violation"] for payload in history_payloads],
        val_violation_runs=[payload["val_violation"] for payload in history_payloads],
    )

    summary_path = multi_seed_dir / "multi_seed_summary.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "case_dir": str(case_dir),
                "dataset_dir": str(dataset_dir),
                "seeds": [int(seed) for seed in seeds],
                "shadow_plot_path": str(shadow_plot_path),
                "runs": run_manifests,
            },
            fh,
            indent=2,
            sort_keys=True,
        )

    print("")
    print(f"[multi-seed] Saved shadow plot: {shadow_plot_path}")
    print(f"[multi-seed] Saved summary: {summary_path}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a testcase multiple times with different config seeds and save a shadow-band summary plot."
    )
    parser.add_argument("case_dir", nargs="?", default=".", help="Path to a standalone test case directory.")
    parser.add_argument("--num-runs", type=int, default=10, help="Number of seeds to run when --seeds is not given.")
    parser.add_argument("--base-seed", type=int, default=None, help="Starting config seed. Defaults to config.json seed.")
    parser.add_argument("--seed-step", type=int, default=1, help="Step between consecutive seeds.")
    parser.add_argument("--seeds", type=str, default=None, help="Comma-separated explicit seeds, e.g. 1,2,3,4")
    args = parser.parse_args(list(argv) if argv is not None else None)

    case_dir = Path(args.case_dir).resolve()
    cfg_dict = _load_json(case_dir / "config.json")
    base_seed = int(cfg_dict["seed"]) if args.base_seed is None else int(args.base_seed)
    seeds = _resolve_seeds(base_seed, args.num_runs, args.seed_step, args.seeds)
    return run_multi_seed_case(case_dir, seeds=seeds)


def main_for_case(case_dir: Path) -> int:
    return main([str(case_dir), *sys.argv[1:]])


if __name__ == "__main__":
    raise SystemExit(main())
