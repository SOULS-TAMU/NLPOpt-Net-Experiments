#!/usr/bin/env python3
from __future__ import annotations

import copy
import csv
import json
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import hashlib
import sys
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

try:
    import torch
except ImportError:
    torch = None

from scripts.plot_utils.plotting import save_shadow_objective_violation_plot  # noqa: E402
from scripts.testcase import unified_runner as unified  # noqa: E402
from scripts.factory.nonconvx_factory import (  # noqa: E402
    DATASET_KIND,
    DC3StyleNonconvexGenerator,
    build_problem_generator,
    build_problem_model,
    build_problem_model_from_data,
    normalize_problem_type,
)
from scripts.misc.optimizer_profile import enrich_optimizer_generation_metadata, history_optimizer_timing_fields  # noqa: E402
from scripts.misc.inequality_multipliers import coerce_ineq_multipliers  # noqa: E402
from scripts.testcase import poly_run as poly_nlpopt  # noqa: E402

jax.config.update("jax_enable_x64", True)

SCHEMA_VERSION = 2
_LOCAL_REQUIRED_KEYS = ("type", "n_y", "n_eq", "n_ineq", "num_samples", "seed", "is_diag_Q")


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
    case_dir = ROOT / "case" / "nonconvx"
    case_dir.mkdir(parents=True, exist_ok=True)
    return case_dir


def _local_paths() -> tuple[Path, Path, Path]:
    case_dir = _case_workspace()
    return case_dir / "data.json", case_dir / "config.json", case_dir / "proj.json"


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)


def _load_local_configs() -> tuple[dict, dict, dict]:
    data_path, cfg_path, proj_path = _local_paths()
    if not data_path.exists() or not cfg_path.exists() or not proj_path.exists():
        raise FileNotFoundError(
            "Expected case/nonconvx/data.json, case/nonconvx/config.json, and case/nonconvx/proj.json."
        )
    return _load_json(data_path), _load_json(cfg_path), _load_json(proj_path)


def _json_hash(payload: dict) -> str:
    return hashlib.sha1(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()[:12]


def _normalize_local_data_cfg(data_cfg: Mapping[str, Any]) -> dict[str, Any]:
    if all(key in data_cfg for key in _LOCAL_REQUIRED_KEYS):
        raw_type = str(data_cfg["type"])
        n_y = int(data_cfg["n_y"])
        n_eq = int(data_cfg["n_eq"])
        n_ineq = int(data_cfg["n_ineq"])
        num_samples = int(data_cfg["num_samples"])
        seed = int(data_cfg["seed"])
        is_diag_q = bool(data_cfg.get("is_diag_Q", True))
        force_regenerate = bool(data_cfg.get("force_regenerate", False))
        schema_version = int(data_cfg.get("schema_version", SCHEMA_VERSION))
    elif all(key in data_cfg for key in ("type", "n", "me", "mi", "num_samples", "seed")):
        raw_type = str(data_cfg["type"])
        n_y = int(data_cfg["n"])
        n_eq = int(data_cfg["me"])
        n_ineq = int(data_cfg["mi"])
        num_samples = int(data_cfg["num_samples"])
        seed = int(data_cfg["seed"])
        is_diag_q = bool(data_cfg.get("is_diag_Q", True))
        force_regenerate = bool(data_cfg.get("force_regenerate", False))
        schema_version = int(data_cfg.get("schema_version", SCHEMA_VERSION))
    else:
        missing = [key for key in _LOCAL_REQUIRED_KEYS if key not in data_cfg]
        raise ValueError(f"case/nonconvx/data.json is missing required keys: {', '.join(sorted(missing))}")

    normalized = {
        "type": normalize_problem_type(raw_type),
        "n_y": n_y,
        "n_eq": n_eq,
        "n_ineq": n_ineq,
        "num_samples": num_samples,
        "seed": seed,
        "is_diag_Q": is_diag_q,
        "force_regenerate": force_regenerate,
        "schema_version": schema_version,
    }
    if normalized["n_y"] <= 0:
        raise ValueError("n_y must be positive.")
    if normalized["n_eq"] < 0 or normalized["n_ineq"] < 0:
        raise ValueError("n_eq and n_ineq must be nonnegative.")
    if normalized["n_eq"] > normalized["n_y"]:
        raise ValueError("n_eq cannot exceed n_y.")
    if normalized["num_samples"] <= 0:
        raise ValueError("num_samples must be positive.")
    return normalized


def _translated_poly_cfg(local_cfg: Mapping[str, Any]) -> dict[str, Any]:
    local = _normalize_local_data_cfg(local_cfg)
    p = int(local["n_eq"])
    return {
        "type": DATASET_KIND,
        "p": p,
        "n": int(local["n_y"]),
        "me": int(local["n_eq"]),
        "mi": int(local["n_ineq"]),
        "num_samples": int(local["num_samples"]),
        "seed": int(local["seed"]),
        "x_L": [-1.0] * p,
        "x_U": [1.0] * p,
        "is_diag_Q": bool(local["is_diag_Q"]),
        "force_regenerate": bool(local.get("force_regenerate", False)),
    }


def _canonical_data_cfg(data_cfg: Mapping[str, Any]) -> dict[str, Any]:
    local = _normalize_local_data_cfg(data_cfg)
    return {
        "type": local["type"],
        "n_y": local["n_y"],
        "n_eq": local["n_eq"],
        "n_ineq": local["n_ineq"],
        "num_samples": local["num_samples"],
        "seed": local["seed"],
        "is_diag_Q": local["is_diag_Q"],
        "schema_version": local["schema_version"],
    }


def build_dataset_id(data_cfg: Mapping[str, Any]) -> str:
    canonical = _canonical_data_cfg(data_cfg)
    stem = (
        f"{canonical['type']}_ny{canonical['n_y']}"
        f"_neq{canonical['n_eq']}_nineq{canonical['n_ineq']}"
        f"_ns{canonical['num_samples']}_seed{canonical['seed']}"
        f"_{'diag' if canonical['is_diag_Q'] else 'dense'}"
    )
    return f"{stem}_{_json_hash(canonical)}"


def dataset_dir(case_dir: Path, data_cfg: Mapping[str, Any]) -> Path:
    return case_dir / "problem_data" / DATASET_KIND / build_dataset_id(data_cfg)


def _paths(base: Path) -> dict[str, Path]:
    return {
        "arrays": base / "dataset.npz",
        "parameters_csv": base / "parameters.csv",
        "variables_csv": base / "variables.csv",
        "ineq_multipliers_csv": base / "ineq_multipliers.csv",
        "metadata": base / "metadata.json",
        "data": base / "data.json",
        "problem_data": base / "problem_data.npz",
    }


def _write_csv(path: Path, arr: np.ndarray) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerows(np.asarray(arr).tolist())


def _dataset_exists(base: Path) -> bool:
    paths = _paths(base)
    return all(
        path.exists()
        for path in (
            paths["arrays"],
            paths["parameters_csv"],
            paths["variables_csv"],
            paths["ineq_multipliers_csv"],
            paths["metadata"],
            paths["data"],
        )
    )


def _save_dataset(
    base: Path,
    *,
    data_cfg: Mapping[str, Any],
    X: np.ndarray,
    Y: np.ndarray,
    Mu: np.ndarray,
    metadata: Mapping[str, Any],
    problem_data: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    base.mkdir(parents=True, exist_ok=True)
    paths = _paths(base)
    np.savez(paths["arrays"], X=np.asarray(X, dtype=np.float64), Y=np.asarray(Y, dtype=np.float64), Mu=np.asarray(Mu, dtype=np.float64))
    _write_csv(paths["parameters_csv"], X)
    _write_csv(paths["variables_csv"], Y)
    _write_csv(paths["ineq_multipliers_csv"], Mu)
    _write_json(paths["data"], _canonical_data_cfg(data_cfg))
    if problem_data is not None:
        np.savez(paths["problem_data"], **{key: np.asarray(value) for key, value in dict(problem_data).items()})
    artifact_paths = [
        paths["arrays"],
        paths["parameters_csv"],
        paths["variables_csv"],
        paths["ineq_multipliers_csv"],
        paths["data"],
    ]
    if problem_data is not None:
        artifact_paths.append(paths["problem_data"])
    enriched_metadata = enrich_optimizer_generation_metadata(
        metadata,
        num_points=int(np.asarray(X).shape[0]),
        artifact_paths=artifact_paths,
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
    generate_fn: Callable[[], tuple],
    *,
    force: bool = False,
) -> DatasetBundle:
    local_cfg = _normalize_local_data_cfg(data_cfg)
    base = dataset_dir(case_dir, local_cfg)
    dataset_id = build_dataset_id(local_cfg)
    if _dataset_exists(base) and not force:
        X, Y, Mu, metadata = _load_dataset(base)
        return DatasetBundle(dataset_dir=base, dataset_id=dataset_id, generated=False, X=X, Y=Y, Mu=Mu, metadata=metadata)

    generated = generate_fn()
    if len(generated) == 3:
        X, Y, metadata = generated
        Mu = np.zeros((int(np.asarray(X).shape[0]), 0), dtype=np.float64)
        problem_data = None
    elif len(generated) == 4:
        X, Y, Mu, metadata = generated
        problem_data = None
    elif len(generated) == 5:
        X, Y, Mu, metadata, problem_data = generated
    else:
        raise ValueError(
            "Dataset generator must return (X, Y, metadata), (X, Y, Mu, metadata), "
            "or (X, Y, Mu, metadata, problem_data)."
        )
    metadata = _save_dataset(base, data_cfg=local_cfg, X=X, Y=Y, Mu=Mu, metadata=metadata, problem_data=problem_data)
    return DatasetBundle(dataset_dir=base, dataset_id=dataset_id, generated=True, X=np.asarray(X), Y=np.asarray(Y), Mu=np.asarray(Mu), metadata=dict(metadata))

def _generate_dataset(generator: DC3StyleNonconvexGenerator, data_cfg: Mapping[str, Any]):
    start_time = time.perf_counter()
    local_cfg = _normalize_local_data_cfg(data_cfg)
    target = int(local_cfg["num_samples"])
    kept_x: list[np.ndarray] = []
    kept_y: list[np.ndarray] = []
    kept_mu: list[np.ndarray] = []
    objectives: list[float] = []
    status_counts: dict[str, int] = {}
    attempts = 0
    max_attempts = max(4 * target, target + 100)

    while len(kept_x) < target and attempts < max_attempts:
        batch_size = min(max(32, target - len(kept_x)), target)
        xs = generator.sample_parameters(batch_size)
        for x in xs:
            result = generator.solve_for_x(x)
            attempts += 1
            status = str(result["status"])
            status_counts[status] = status_counts.get(status, 0) + 1
            if status == "optimal" and result["y"] is not None:
                kept_x.append(np.asarray(x, dtype=np.float64))
                kept_y.append(np.asarray(result["y"], dtype=np.float64))
                kept_mu.append(coerce_ineq_multipliers(result.get("mu"), generator.n_ineq))
                objectives.append(float(result["objective"]) if result["objective"] is not None else np.nan)
                if len(kept_x) >= target:
                    break
        if len(kept_x) >= target:
            break

    if len(kept_x) < target:
        raise RuntimeError(
            f"Only collected {len(kept_x)} successful nonconvex points out of requested {target}. "
            "Increase num_samples or improve solver robustness."
        )

    metadata = {
        "problem_type": DATASET_KIND,
        "n_x": int(generator.n_x),
        "n_y": int(generator.n_y),
        "n_eq": int(generator.n_eq),
        "n_ineq": int(generator.n_ineq),
        "num_samples": target,
        "seed": int(local_cfg["seed"]),
        "solver": str(generator.requested_solver),
        "is_diag_Q": bool(local_cfg["is_diag_Q"]),
        "objective_min": float(np.nanmin(objectives)) if objectives else np.nan,
        "objective_max": float(np.nanmax(objectives)) if objectives else np.nan,
        "objective_mean": float(np.nanmean(objectives)) if objectives else np.nan,
        "status_counts": status_counts,
        "attempts": int(attempts),
        "optimizer_generation_wall_time_sec": time.perf_counter() - start_time,
    }
    return (
        np.asarray(kept_x, dtype=np.float64),
        np.asarray(kept_y, dtype=np.float64),
        np.stack(kept_mu, axis=0) if kept_mu else np.zeros((len(kept_x), generator.n_ineq), dtype=np.float64),
        metadata,
        generator.get_problem_data(),
    )


class DC3StyleNonconvexProblem:
    def __init__(
        self,
        problem_data: Mapping[str, Any],
        X: np.ndarray,
        Y: np.ndarray,
        *,
        train_frac: float,
        valid_frac: float,
        test_frac: float,
        split_seed: int,
    ) -> None:
        if torch is None:
            raise RuntimeError("Torch is required for DC3-style nonconvex problem wrappers.")

        self._Q = torch.tensor(np.asarray(problem_data["Q"]), dtype=torch.float64)
        self._p = torch.tensor(np.asarray(problem_data["p"]), dtype=torch.float64)
        self._A = torch.tensor(np.asarray(problem_data["A"]), dtype=torch.float64)
        self._G = torch.tensor(np.asarray(problem_data["G"]), dtype=torch.float64)
        self._h = torch.tensor(np.asarray(problem_data["h"]), dtype=torch.float64)
        self._X_all = torch.tensor(np.asarray(X), dtype=torch.float64)
        self._Y_all = torch.tensor(np.asarray(Y), dtype=torch.float64)
        self._xdim = int(self._X_all.shape[1])
        self._ydim = int(self._Y_all.shape[1])
        self._neq = int(self._A.shape[0])
        self._nineq = int(self._G.shape[0])
        self._nknowns = 0
        self._use_completion = True
        self._device = self._X_all.device
        self._valid_frac = float(valid_frac)
        self._test_frac = float(test_frac)
        self._train_frac = float(train_frac)

        perm = np.random.default_rng(int(split_seed)).permutation(int(self._X_all.shape[0]))
        self._X_all = self._X_all[perm]
        self._Y_all = self._Y_all[perm]
        self._num = int(self._X_all.shape[0])

        n_train = int(self._num * self._train_frac)
        n_valid = int(self._num * (self._train_frac + self._valid_frac)) - n_train
        self._trainX = self._X_all[:n_train]
        self._validX = self._X_all[n_train:n_train + n_valid]
        self._testX = self._X_all[n_train + n_valid:]
        self._trainY = self._Y_all[:n_train]
        self._validY = self._Y_all[n_train:n_train + n_valid]
        self._testY = self._Y_all[n_train + n_valid:]

        det = 0.0
        attempts = 0
        chooser = np.random.default_rng(int(split_seed) + 17)
        while abs(det) < 1e-4 and attempts < 100:
            self._partial_vars = chooser.choice(self._ydim, self._ydim - self._neq, replace=False)
            self._other_vars = np.setdiff1d(np.arange(self._ydim), self._partial_vars)
            det = float(torch.det(self._A[:, self._other_vars]).item())
            attempts += 1
        if attempts >= 100:
            raise RuntimeError("Could not construct a stable completion split for the nonconvex DC3 problem.")
        self._A_partial = self._A[:, self._partial_vars]
        self._A_other_inv = torch.inverse(self._A[:, self._other_vars])
        self._M = 2 * (self.G[:, self.partial_vars] - self.G[:, self.other_vars] @ (self._A_other_inv @ self._A_partial))

    @property
    def Q(self):
        return self._Q

    @property
    def p(self):
        return self._p

    @property
    def A(self):
        return self._A

    @property
    def G(self):
        return self._G

    @property
    def h(self):
        return self._h

    @property
    def xdim(self):
        return self._xdim

    @property
    def ydim(self):
        return self._ydim

    @property
    def neq(self):
        return self._neq

    @property
    def nineq(self):
        return self._nineq

    @property
    def nknowns(self):
        return self._nknowns

    @property
    def partial_vars(self):
        return self._partial_vars

    @property
    def other_vars(self):
        return self._other_vars

    @property
    def partial_unknown_vars(self):
        return self._partial_vars

    @property
    def trainX(self):
        return self._trainX

    @property
    def validX(self):
        return self._validX

    @property
    def testX(self):
        return self._testX

    @property
    def trainY(self):
        return self._trainY

    @property
    def validY(self):
        return self._validY

    @property
    def testY(self):
        return self._testY

    @property
    def device(self):
        return self._device

    def to(self, device):
        for attr in ("_Q", "_p", "_A", "_G", "_h", "_X_all", "_Y_all", "_A_partial", "_A_other_inv", "_M", "_trainX", "_validX", "_testX", "_trainY", "_validY", "_testY"):
            setattr(self, attr, getattr(self, attr).to(device))
        self._device = device
        return self

    def obj_fn(self, Y):
        quad = 0.5 * torch.sum((Y @ self.Q) * Y, dim=1)
        nonlinear = torch.sum(self.p * torch.sin(Y), dim=1)
        return quad + nonlinear

    def eq_resid(self, X, Y):
        return X - Y @ self.A.T

    def ineq_resid(self, X, Y):
        del X
        return Y @ self.G.T - self.h

    def _nonlinear_resid(self, X, Y):
        return self.ineq_resid(X, Y)

    def ineq_dist(self, X, Y):
        return torch.clamp(self.ineq_resid(X, Y), min=0.0)

    def eq_grad(self, X, Y):
        return 2 * (Y @ self.A.T - X) @ self.A

    def ineq_grad(self, X, Y):
        return 2 * torch.clamp(Y @ self.G.T - self.h, min=0.0) @ self.G

    def ineq_partial_grad(self, X, Y):
        del X
        grad = torch.clamp(Y @ self.G.T - self.h, min=0.0) @ self._M
        out = torch.zeros(Y.shape[0], self.ydim, dtype=Y.dtype, device=Y.device)
        out[:, self.partial_vars] = grad
        out[:, self.other_vars] = -(grad @ self._A_partial.T) @ self._A_other_inv.T
        return out

    def process_output(self, X, Y):
        del X
        return Y

    def complete_partial(self, X, Z):
        Y = torch.zeros(X.shape[0], self.ydim, dtype=Z.dtype, device=Z.device)
        Y[:, self.partial_vars] = Z
        Y[:, self.other_vars] = (X - Z @ self._A_partial.T) @ self._A_other_inv.T
        return Y

    def _bounds(self, X):
        lower = -torch.inf * torch.ones((X.shape[0], self.ydim), dtype=X.dtype, device=X.device)
        upper = torch.inf * torch.ones((X.shape[0], self.ydim), dtype=X.dtype, device=X.device)
        return lower, upper


def _resolve_split_fracs(cfg_dict: Mapping[str, Any]) -> tuple[float, float, float]:
    train_frac = float(cfg_dict.get("train_frac", 0.8))
    if not 0.0 < train_frac < 1.0:
        raise ValueError("config.json must satisfy 0 < train_frac < 1 for case/nonconvx.")
    holdout = max(0.0, 1.0 - train_frac)
    requested_val = float(cfg_dict.get("val_frac", holdout / 2.0))
    valid_frac = min(max(requested_val, 0.0), holdout)
    test_frac = max(0.0, holdout - valid_frac)
    if holdout > 0.0 and test_frac == 0.0:
        valid_frac = holdout / 2.0
        test_frac = holdout - valid_frac
    if valid_frac <= 0.0 or test_frac <= 0.0:
        raise ValueError(
            "case/nonconvx needs non-empty validation and test splits. "
            "Use train_frac < 1.0 so the holdout can be split across both."
        )
    return train_frac, valid_frac, test_frac


def _prepare_problem(case_dir: Path, data_cfg: Mapping[str, Any], cfg_dict: Mapping[str, Any]):
    local_cfg = _normalize_local_data_cfg(data_cfg)
    generator = build_problem_generator(local_cfg)
    dataset = ensure_cached_dataset(
        case_dir,
        local_cfg,
        lambda: _generate_dataset(generator, local_cfg),
        force=bool(local_cfg.get("force_regenerate", False)),
    )
    train_frac, valid_frac, test_frac = _resolve_split_fracs(cfg_dict)
    problem = DC3StyleNonconvexProblem(
        generator.get_problem_data(),
        dataset.X,
        dataset.Y,
        train_frac=train_frac,
        valid_frac=valid_frac,
        test_frac=test_frac,
        split_seed=int(cfg_dict.get("seed", 42)),
    )
    return dataset, problem


def _build_local_dc3_args(cfg_dict: Mapping[str, Any], problem, *, use_completion: bool) -> dict:
    defaults = unified.dc3_default_args.method_default_args("nonconvex")
    corr_mode = str(cfg_dict.get("dc3_corrMode", defaults["corrMode"])).lower()
    if corr_mode == "partial" and not use_completion:
        corr_mode = "full"
    return {
        "probType": "nonconvex",
        "epochs": int(cfg_dict.get("epochs", defaults["epochs"])),
        "printEvery": max(1, int(cfg_dict.get("print_every", 10))),
        "batchSize": int(cfg_dict.get("batch_size", defaults["batchSize"])),
        "lr": float(cfg_dict.get("learning_rate", defaults["lr"])),
        "hiddenSize": int(cfg_dict.get("hidden_size", defaults["hiddenSize"])),
        "softWeight": float(cfg_dict.get("dc3_softWeight", cfg_dict.get("alpha_consistency", defaults["softWeight"]))),
        "softWeightEqFrac": float(cfg_dict.get("dc3_softWeightEqFrac", defaults["softWeightEqFrac"])),
        "useCompl": bool(use_completion),
        "useTrainCorr": bool(cfg_dict.get("dc3_useTrainCorr", defaults["useTrainCorr"])),
        "useTestCorr": bool(cfg_dict.get("dc3_useTestCorr", defaults["useTestCorr"])),
        "corrMode": corr_mode,
        "corrTrainSteps": int(cfg_dict.get("dc3_corrTrainSteps", defaults["corrTrainSteps"])),
        "corrTestMaxSteps": int(cfg_dict.get("dc3_corrTestMaxSteps", defaults["corrTestMaxSteps"])),
        "corrEps": float(cfg_dict.get("dc3_corrEps", defaults["corrEps"])),
        "corrLr": float(cfg_dict.get("dc3_corrLr", defaults["corrLr"])),
        "corrMomentum": float(cfg_dict.get("dc3_corrMomentum", defaults["corrMomentum"])),
        "saveAllStats": bool(cfg_dict.get("dc3_saveAllStats", defaults["saveAllStats"])),
        "resultsSaveFreq": int(cfg_dict.get("dc3_resultsSaveFreq", defaults["resultsSaveFreq"])),
        "seed": int(cfg_dict.get("seed", 42)),
        "nonconvexVar": int(problem.ydim),
        "nonconvexIneq": int(problem.nineq),
        "nonconvexEq": int(problem.neq),
        "nonconvexEx": int(problem.trainX.shape[0] + problem.validX.shape[0] + problem.testX.shape[0]),
    }


def _build_local_baseline_nn_args(cfg_dict: Mapping[str, Any], problem) -> dict:
    args = unified._build_baseline_nn_args(dict(cfg_dict))
    args["probType"] = "nonconvex"
    args["nonconvexVar"] = int(problem.ydim)
    args["nonconvexIneq"] = int(problem.nineq)
    args["nonconvexEq"] = int(problem.neq)
    args["nonconvexEx"] = int(problem.trainX.shape[0] + problem.validX.shape[0] + problem.testX.shape[0])
    return args


def _build_local_baseline_eq_nn_args(cfg_dict: Mapping[str, Any], problem, *, use_completion: bool) -> dict:
    args = unified._build_baseline_eq_nn_args(dict(cfg_dict), use_completion=use_completion)
    args["probType"] = "nonconvex"
    args["nonconvexVar"] = int(problem.ydim)
    args["nonconvexIneq"] = int(problem.nineq)
    args["nonconvexEq"] = int(problem.neq)
    args["nonconvexEx"] = int(problem.trainX.shape[0] + problem.validX.shape[0] + problem.testX.shape[0])
    return args


@contextmanager
def _patch_unified_for_nonconvex():
    original_problem_family = unified._problem_family
    original_filtered_data_cfg = unified._filtered_data_cfg
    original_prepare_poly_problem = unified._prepare_poly_problem
    original_resolve_dataset_dir = unified._resolve_dataset_dir
    original_case_workspace = unified._case_workspace
    original_build_dc3_args = unified.poly_torch._build_dc3_args

    def patched_problem_family(data_cfg: Mapping[str, Any]) -> str:
        if str(data_cfg.get("type", "")).strip().lower() == DATASET_KIND:
            return "poly"
        return original_problem_family(data_cfg)

    def patched_filtered_data_cfg(data_cfg: Mapping[str, Any]):
        if str(data_cfg.get("type", "")).strip().lower() == DATASET_KIND:
            return _normalize_local_data_cfg(data_cfg)
        return original_filtered_data_cfg(data_cfg)

    def patched_prepare_poly_problem(case_dir: Path, data_cfg: Mapping[str, Any], cfg_dict: Mapping[str, Any]):
        if str(data_cfg.get("type", "")).strip().lower() == DATASET_KIND:
            return _prepare_problem(case_dir, data_cfg, cfg_dict)
        return original_prepare_poly_problem(case_dir, data_cfg, cfg_dict)

    def patched_resolve_dataset_dir(case_dir: Path, data_cfg: Mapping[str, Any]):
        if str(data_cfg.get("type", "")).strip().lower() == DATASET_KIND:
            return dataset_dir(case_dir, data_cfg)
        return original_resolve_dataset_dir(case_dir, data_cfg)

    def patched_build_dc3_args(cfg_dict: Mapping[str, Any], data_cfg: Mapping[str, Any], *, use_completion: bool):
        if str(data_cfg.get("type", "")).strip().lower() == DATASET_KIND:
            _dataset, problem = _prepare_problem(_case_workspace(), data_cfg, cfg_dict)
            return _build_local_dc3_args(cfg_dict, problem, use_completion=use_completion)
        return original_build_dc3_args(cfg_dict, data_cfg, use_completion=use_completion)

    unified._problem_family = patched_problem_family
    unified._filtered_data_cfg = patched_filtered_data_cfg
    unified._prepare_poly_problem = patched_prepare_poly_problem
    unified._resolve_dataset_dir = patched_resolve_dataset_dir
    unified._case_workspace = _case_workspace
    unified.poly_torch._build_dc3_args = patched_build_dc3_args
    try:
        yield
    finally:
        unified._problem_family = original_problem_family
        unified._filtered_data_cfg = original_filtered_data_cfg
        unified._prepare_poly_problem = original_prepare_poly_problem
        unified._resolve_dataset_dir = original_resolve_dataset_dir
        unified._case_workspace = original_case_workspace
        unified.poly_torch._build_dc3_args = original_build_dc3_args


@contextmanager
def _patch_poly_nlpopt_for_nonconvex():
    original_normalize_problem_type = poly_nlpopt.normalize_problem_type
    original_build_problem_generator = poly_nlpopt.build_problem_generator
    original_build_problem_model = poly_nlpopt.build_problem_model
    original_build_problem_model_from_data = poly_nlpopt.build_problem_model_from_data
    original_uses_nonconvex_generator = poly_nlpopt.uses_nonconvex_generator
    original_ensure_cached_dataset = poly_nlpopt.ensure_cached_dataset

    poly_nlpopt.normalize_problem_type = normalize_problem_type
    poly_nlpopt.build_problem_generator = build_problem_generator
    poly_nlpopt.build_problem_model = lambda data_cfg, dtype=jnp.float64: build_problem_model(data_cfg, dtype=dtype)
    poly_nlpopt.build_problem_model_from_data = lambda problem_data, dtype=jnp.float64: build_problem_model_from_data(problem_data, dtype=dtype)
    poly_nlpopt.uses_nonconvex_generator = lambda _data_cfg: True
    poly_nlpopt.ensure_cached_dataset = ensure_cached_dataset
    try:
        yield
    finally:
        poly_nlpopt.normalize_problem_type = original_normalize_problem_type
        poly_nlpopt.build_problem_generator = original_build_problem_generator
        poly_nlpopt.build_problem_model = original_build_problem_model
        poly_nlpopt.build_problem_model_from_data = original_build_problem_model_from_data
        poly_nlpopt.uses_nonconvex_generator = original_uses_nonconvex_generator
        poly_nlpopt.ensure_cached_dataset = original_ensure_cached_dataset


def _run_nlpopt(
    case_dir: Path,
    data_cfg: dict,
    cfg_dict: dict,
    proj_cfg: dict,
    *,
    output_dir: Path,
) -> unified.RunArtifacts:
    translated_cfg = _translated_poly_cfg(data_cfg)
    with _patch_poly_nlpopt_for_nonconvex():
        poly_nlpopt.run_case(
            case_dir,
            data_cfg_override=translated_cfg,
            cfg_dict_override=unified._training_cfg_only(cfg_dict),
            proj_cfg_override=proj_cfg,
            output_dir_override=output_dir,
        )
    dataset_root = dataset_dir(case_dir, data_cfg)
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


def _run_torch_framework(
    case_dir: Path,
    data_cfg: dict,
    cfg_dict: dict,
    proj_cfg: dict,
    framework: str,
    *,
    output_dir: Path,
) -> unified.RunArtifacts:
    if torch is None:
        raise RuntimeError("Torch is required for DC3, Baseline NN, and Baseline Eq. NN runs.")

    dataset, problem = _prepare_problem(case_dir, data_cfg, cfg_dict)
    save_dir = Path(output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if framework == "dc3":
        module = unified.poly_torch._load_dc3_method_module()
        args = _build_local_dc3_args(cfg_dict, problem, use_completion=bool(problem._use_completion))
        predict_fn = lambda solver_net, X: unified._predict_dc3(module, problem, solver_net, args, X)
        raw_predict_fn = lambda solver_net, X: unified._predict_dc3_raw(module, problem, solver_net, args, X)
        trainer = lambda: unified.poly_torch._train_net_with_nlpopt_logging(module, problem, args, str(save_dir))
    elif framework == "baseline_nn":
        module = unified._load_dc3_runtime_module("baseline_nn.py", "_nlpopt_dc3_baseline_nn_runtime")
        args = _build_local_baseline_nn_args(cfg_dict, problem)
        predict_fn = lambda solver_net, X: unified._predict_baseline_nn(module, problem, solver_net, args, X)
        raw_predict_fn = lambda solver_net, X: unified._predict_baseline_nn_raw(module, problem, solver_net, args, X)
        trainer = lambda: unified._train_baseline_nn_with_logging(module, problem, args, save_dir)
    elif framework == "baseline_eq_nn":
        module = unified._load_dc3_runtime_module("baseline_eq_nn.py", "_nlpopt_dc3_baseline_eq_nn_runtime")
        args = _build_local_baseline_eq_nn_args(cfg_dict, problem, use_completion=bool(problem._use_completion))
        predict_fn = lambda solver_net, X: unified._predict_baseline_eq_nn(module, problem, solver_net, args, X)
        raw_predict_fn = lambda solver_net, X: unified._predict_baseline_eq_nn_raw(module, problem, solver_net, args, X)
        trainer = lambda: unified._train_baseline_eq_nn_with_logging(module, problem, args, save_dir)
    else:
        raise ValueError(f"Unsupported framework '{framework}'.")

    problem.to(module.DEVICE)

    unified._write_json(save_dir / "args.json", args)
    with open(save_dir / "args.dict", "wb") as fh:
        import pickle

        pickle.dump(args, fh)
    unified._write_run_configs(save_dir, data_cfg, cfg_dict, proj_cfg)

    print(f"Dataset: {dataset.dataset_dir}")
    print(f"Dataset status: {'generated' if dataset.generated else 'reused'}")
    print(f"{unified._framework_label(framework)} save dir: {save_dir}")
    print(
        f"NONCONVEX  n_x={int(problem.xdim)} n_y={int(problem.ydim)} "
        f"n_eq={int(problem.neq)} n_ineq={int(problem.nineq)}"
    )
    print(f"Device: {module.DEVICE}")
    print(
        f"batch_size={int(args['batchSize'])}  "
        f"train_batches={unified._num_batches(problem.trainX.shape[0], int(args['batchSize']))}  "
        f"val_batches={unified._num_batches(problem.validX.shape[0], int(args['batchSize']))}  "
        f"test_batches={unified._num_batches(problem.testX.shape[0], int(args['batchSize']))}"
    )
    print(f"Splits: train={problem.trainX.shape[0]}  valid={problem.validX.shape[0]}  test={problem.testX.shape[0]}")

    solver_net, stats, profile = trainer()
    history_path = unified._save_history_and_plot(
        save_dir,
        framework,
        stats,
        problem,
        extra_history=history_optimizer_timing_fields(dataset.metadata),
    )
    metrics = unified._print_final_supervised_summary(problem, lambda X: predict_fn(solver_net, X), profile, predict_raw_fn=lambda X: raw_predict_fn(solver_net, X))
    summary = unified._summarize_stats(stats)
    train_count = int(problem.trainX.shape[0])
    val_count = int(problem.validX.shape[0])
    summary.update(metrics)
    summary["objective_value"] = unified._weighted_final_mean(stats, "train_eval", "valid_eval", train_count=train_count, val_count=val_count)
    summary["max_equality"] = unified._combined_final_max(stats, "train_eq_max", "valid_eq_max")
    summary["mean_equality"] = unified._weighted_final_mean(stats, "train_eq_mean", "valid_eq_mean", train_count=train_count, val_count=val_count)
    summary["max_inequality"] = unified._combined_final_max(stats, "train_ineq_max", "valid_ineq_max")
    summary["mean_inequality"] = unified._weighted_final_mean(stats, "train_ineq_mean", "valid_ineq_mean", train_count=train_count, val_count=val_count)
    summary["consistency"] = unified._weighted_final_mean(stats, "train_dist", "valid_dist", train_count=train_count, val_count=val_count)
    summary["optimality_gap"] = float(summary.get("relative_objective_gap", float("nan")))
    summary["dataset_dir"] = str(dataset.dataset_dir)
    summary["save_dir"] = str(save_dir)
    summary["framework"] = framework
    summary["space_mb"] = unified._directory_size_mb(save_dir)
    summary_path = save_dir / "summary.json"
    _write_json(summary_path, summary)
    print(f"[{framework}] Saved: {save_dir}")

    return unified.RunArtifacts(
        framework=framework,
        dataset_dir=dataset.dataset_dir,
        run_dir=save_dir,
        history_path=history_path,
        metrics_path=summary_path,
        plot_path=save_dir / "compare_metrics.png",
    )


def _print_run_header(case_dir: Path, data_cfg: dict, cfg_dict: dict, framework: str) -> None:
    dataset_target = dataset_dir(case_dir, data_cfg)
    print("=" * 80)
    print(f"Standalone runner | framework={unified._framework_label(framework)}")
    print(
        f"NONCONVEX  n_x={int(data_cfg['n_eq'])} n_y={int(data_cfg['n_y'])} "
        f"n_eq={int(data_cfg['n_eq'])} n_ineq={int(data_cfg['n_ineq'])}"
    )
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
    *,
    output_dir_override: Optional[Path] = None,
) -> unified.RunArtifacts:
    data_cfg = _normalize_local_data_cfg(data_cfg)
    framework = unified._normalize_model_name(str(cfg_dict.get("model", "nlpopt")))
    _print_run_header(case_dir, data_cfg, cfg_dict, framework)
    dataset_root = dataset_dir(case_dir, data_cfg)
    output_dir = Path(output_dir_override) if output_dir_override is not None else unified._framework_dir(dataset_root, framework)
    if framework == "nlpopt":
        return _run_nlpopt(case_dir, data_cfg, cfg_dict, proj_cfg, output_dir=output_dir)
    return _run_torch_framework(case_dir, data_cfg, cfg_dict, proj_cfg, framework, output_dir=output_dir)


def _run_multi_seed(
    case_dir: Path,
    data_cfg: dict,
    cfg_dict: dict,
    proj_cfg: dict,
    *,
    multi_dir_override: Optional[Path] = None,
) -> int:
    data_cfg = _normalize_local_data_cfg(data_cfg)
    framework = unified._normalize_model_name(str(cfg_dict.get("model", "nlpopt")))
    base_seed = int(cfg_dict.get("seed", 42))
    num_seeds = int(cfg_dict.get("num_seeds", 10))
    if num_seeds <= 0:
        raise ValueError("num_seeds must be positive when run_multiple_seed is true.")
    seeds = [base_seed + idx for idx in range(num_seeds)]

    dataset_root = dataset_dir(case_dir, data_cfg)
    multi_dir = Path(multi_dir_override) if multi_dir_override is not None else unified._framework_multi_dir(dataset_root, framework)
    multi_dir.mkdir(parents=True, exist_ok=True)
    unified._write_run_configs(multi_dir, data_cfg, cfg_dict, proj_cfg)

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

    shadow_plot_path = None
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
            series_label=unified._framework_label(framework),
        )

    summary_path = multi_dir / "multi_seed_summary.json"
    _write_json(
        summary_path,
        {
            "framework": framework,
            "framework_label": unified._framework_label(framework),
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
        data_cfg=data_cfg,
        cfg_dict=cfg_dict,
        proj_cfg=proj_cfg,
        framework=framework,
        seeds=seeds,
        extra={"summary_path": str(summary_path)},
    )
    print(f"[multi-seed] Saved summary: {summary_path}")
    print(f"[multi-seed] Updated family metadata: {metadata_path}")
    return 0


def default_dataset_dir() -> Path:
    data_cfg, _cfg, _proj = _load_local_configs()
    return dataset_dir(_case_workspace(), data_cfg)


def main() -> int:
    data_cfg, cfg_dict, proj_cfg = _load_local_configs()
    data_cfg = _normalize_local_data_cfg(data_cfg)
    case_dir = _case_workspace()

    if unified._str_to_bool(cfg_dict.get("run_multiple_seed", False)):
        return _run_multi_seed(case_dir, data_cfg, cfg_dict, proj_cfg)

    artifacts = _run_single_case(case_dir, data_cfg, cfg_dict, proj_cfg)
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


if __name__ == "__main__":
    raise SystemExit(main())
