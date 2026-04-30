#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from scripts.misc.cli_overrides import apply_cli_overrides
from scripts.testcase import nonconvx_run as base_runner


def load_local_configs(case_dir: Path) -> tuple[dict, dict, dict]:
    data_cfg = base_runner._load_json(case_dir / "data.json")
    cfg_dict = base_runner._load_json(case_dir / "config.json")
    proj_cfg = base_runner._load_json(case_dir / "proj.json")
    data_cfg, cfg_dict = apply_cli_overrides(data_cfg, cfg_dict)
    return data_cfg, cfg_dict, proj_cfg


def normalize_local_data_cfg(data_cfg: dict) -> dict:
    required = ("type", "p", "n", "me", "mi", "num_samples", "seed")
    missing = [key for key in required if key not in data_cfg]
    if missing:
        raise ValueError(
            "case/nonconvx/data.json is missing required keys: "
            + ", ".join(sorted(missing))
        )

    p = int(data_cfg["p"])
    me = int(data_cfg["me"])
    if p != me:
        raise ValueError(
            "case/nonconvx currently requires p == me because the DC3-style "
            "nonconvex family uses one parameter per equality constraint."
        )

    translated = {
        "type": data_cfg["type"],
        "n": int(data_cfg["n"]),
        "me": me,
        "mi": int(data_cfg["mi"]),
        "num_samples": int(data_cfg["num_samples"]),
        "seed": int(data_cfg["seed"]),
        "is_diag_Q": bool(data_cfg.get("is_diag_Q", True)),
        "force_regenerate": bool(data_cfg.get("force_regenerate", False)),
        "schema_version": int(data_cfg.get("schema_version", base_runner.SCHEMA_VERSION)),
    }
    return base_runner._normalize_local_data_cfg(translated)


def default_dataset_dir(case_dir: Path) -> Path:
    data_cfg, _cfg_dict, _proj_cfg = load_local_configs(case_dir)
    return base_runner.dataset_dir(case_dir, normalize_local_data_cfg(data_cfg))


def default_comparison_dir(case_dir: Path) -> Path:
    return default_dataset_dir(case_dir) / "comparison"


def run_case(case_dir: Path, _path_arg: str | None = None) -> int:
    raw_data_cfg, cfg_dict, proj_cfg = load_local_configs(case_dir)
    data_cfg = normalize_local_data_cfg(raw_data_cfg)

    if base_runner.unified._str_to_bool(cfg_dict.get("run_multiple_seed", False)):
        return base_runner._run_multi_seed(case_dir, data_cfg, cfg_dict, proj_cfg)

    artifacts = base_runner._run_single_case(case_dir, data_cfg, cfg_dict, proj_cfg)
    metadata_path = base_runner.unified._append_family_metadata(
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
