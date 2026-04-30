#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from scripts.misc.cli_overrides import apply_cli_overrides
from scripts.testcase import unified_runner as unified


def load_local_configs(case_dir: Path) -> tuple[dict, dict, dict]:
    data_cfg = unified._load_json(case_dir / "data.json")
    cfg_dict = unified._load_json(case_dir / "config.json")
    proj_cfg = unified._load_json(case_dir / "proj.json")
    data_cfg, cfg_dict = apply_cli_overrides(data_cfg, cfg_dict)
    return data_cfg, cfg_dict, proj_cfg


def default_dataset_dir(case_dir: Path) -> Path:
    data_cfg, _cfg_dict, _proj_cfg = load_local_configs(case_dir)
    return unified._resolve_dataset_dir(case_dir, data_cfg)


def default_comparison_dir(case_dir: Path) -> Path:
    return default_dataset_dir(case_dir) / "comparison"


def run_case(case_dir: Path, _path_arg: str | None = None) -> int:
    data_cfg, cfg_dict, proj_cfg = load_local_configs(case_dir)

    if unified._str_to_bool(cfg_dict.get("run_multiple_seed", False)):
        return unified._run_multi_seed(case_dir, data_cfg, cfg_dict, proj_cfg)

    artifacts = unified._run_single_case(case_dir, data_cfg, cfg_dict, proj_cfg)
    metadata_path = unified._append_family_metadata(
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
