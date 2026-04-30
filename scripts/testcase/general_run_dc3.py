#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import pickle
import sys
import time
from pathlib import Path

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

from case.general import build_problem_generator, build_problem_model, default_case_dir  # noqa: E402
from case.general.factory import load_model_module, model_definition_path  # noqa: E402
from scripts.misc.runtime_seed import seed_torch_runtime  # noqa: E402
from scripts.misc.training_timing import should_track_epoch, summarize_timing_profile, timing_window_label  # noqa: E402
from scripts.plot_utils.plotting import (  # noqa: E402
    save_objective_value_violation_plot,
    save_shadow_objective_value_violation_plot,
)
from scripts.testcase import general_run as general_local  # noqa: E402
from scripts.testcase import unified_runner as unified  # noqa: E402
from scripts.testcase.nlp_run_dc3 import (  # noqa: E402
    _build_dc3_args,
    _dc3_consistency,
    _dc3_violation,
    _load_dc3_method_module,
    _max_stat,
    _mean_stat,
    _resolve_split_fracs,
    _save_dc3_training_artifacts,
    _save_dir,
    _summarize_stats,
)


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _write_json(path: Path, payload: dict) -> None:
    unified._write_json(path, payload)


def _combine_affine_constraints(model, *, kind: str, param_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_x = int(model.parameter_spec.shapes[param_name][0])
    mats = []
    rhs_consts = []
    param_mats = []

    for entry in model.constraints:
        if entry.kind != kind:
            continue
        if entry.structure != "affine":
            raise ValueError(f"DC3 general runner only supports affine {kind} constraints. Found '{entry.structure}'.")
        meta = entry.metadata or {}
        jac = np.asarray(meta["jac_matrix"], dtype=np.float64)
        rhs = np.asarray(meta.get("rhs_const", 0.0), dtype=np.float64).reshape(-1)
        if rhs.size == 1 and jac.shape[0] != 1:
            rhs = np.full((jac.shape[0],), float(rhs[0]), dtype=np.float64)
        elif rhs.shape != (jac.shape[0],):
            raise ValueError(f"Affine {kind} rhs has shape {rhs.shape}, expected ({jac.shape[0]},).")
        param_mat = np.zeros((jac.shape[0], n_x), dtype=np.float64)
        for block, x_name in meta.get("x_blocks", []):
            if x_name != param_name:
                continue
            param_mat += np.asarray(block, dtype=np.float64)
        mats.append(jac)
        rhs_consts.append(rhs)
        param_mats.append(param_mat)

    if not mats:
        return (
            np.zeros((0, model.var_spec.total_size), dtype=np.float64),
            np.zeros((0,), dtype=np.float64),
            np.zeros((0, n_x), dtype=np.float64),
        )

    return (
        np.vstack(mats),
        np.concatenate(rhs_consts),
        np.vstack(param_mats),
    )


def _extract_affine_bounds(model, *, var_name: str = "y") -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lower_fun = model.bounds.lower_fun
    upper_fun = model.bounds.upper_fun
    if lower_fun is None or upper_fun is None:
        raise ValueError("DC3 general runner expects both lower and upper affine bounds.")
    if not lower_fun.__closure__ or not upper_fun.__closure__:
        raise ValueError("Unable to recover affine bound data from bound closures.")

    lower_builder = lower_fun.__closure__[0].cell_contents
    upper_builder = upper_fun.__closure__[0].cell_contents
    lower_bound = lower_builder._lower_bounds[var_name]
    upper_bound = upper_builder._upper_bounds[var_name]
    return (
        np.asarray(lower_bound.M, dtype=np.float64),
        np.asarray(lower_bound.c, dtype=np.float64),
        np.asarray(upper_bound.M, dtype=np.float64),
        np.asarray(upper_bound.c, dtype=np.float64),
    )


class _JaxObjectiveBridge:
    def __init__(self, model, *, param_name: str):
        self._dtype = model.dtype
        self._param_name = param_name

        def value_single(x, y):
            return model.objective_value({param_name: x}, y)

        def grad_single(x, y):
            return jax.grad(value_single, argnums=1)(x, y)

        self._value_batch = jax.jit(jax.vmap(value_single, in_axes=(0, 0)))
        self._grad_batch = jax.jit(jax.vmap(grad_single, in_axes=(0, 0)))

    def value_and_grad(self, x_np: np.ndarray, y_np: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = jnp.asarray(x_np, dtype=self._dtype)
        y = jnp.asarray(y_np, dtype=self._dtype)
        values = np.array(self._value_batch(x, y), dtype=np.float64, copy=True)
        grads = np.array(self._grad_batch(x, y), dtype=np.float64, copy=True)
        return values, grads


class _TorchJaxObjective(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_t, y_t, bridge):
        values, grads = bridge.value_and_grad(
            x_t.detach().cpu().numpy(),
            y_t.detach().cpu().numpy(),
        )
        grad_y = torch.as_tensor(grads, dtype=y_t.dtype, device=y_t.device)
        ctx.save_for_backward(grad_y)
        return torch.as_tensor(values, dtype=y_t.dtype, device=y_t.device)

    @staticmethod
    def backward(ctx, grad_output):
        (grad_y,) = ctx.saved_tensors
        return None, grad_output.unsqueeze(1) * grad_y, None


class GeneralDC3Problem:
    def __init__(
        self,
        *,
        model,
        param_name: str,
        module,
        X: np.ndarray,
        train_frac: float,
        valid_frac: float,
        test_frac: float,
        split_seed: int,
    ):
        if torch is None:
            raise RuntimeError("DC3 runner requires `torch`. Install PyTorch in your environment first.")

        rng = np.random.default_rng(split_seed)
        order = np.arange(X.shape[0])
        rng.shuffle(order)
        X = np.asarray(X[order], dtype=np.float64)

        A, b, B = _combine_affine_constraints(model, kind="eq", param_name=param_name)
        C, d, E = _combine_affine_constraints(model, kind="ineq", param_name=param_name)
        lower_M, lower_c, upper_M, upper_c = _extract_affine_bounds(model)

        self._X = torch.tensor(X, dtype=torch.float64)
        self._A = torch.tensor(A, dtype=torch.float64)
        self._b = torch.tensor(b, dtype=torch.float64)
        self._B = torch.tensor(B, dtype=torch.float64)
        self._C = torch.tensor(C, dtype=torch.float64)
        self._d = torch.tensor(d, dtype=torch.float64)
        self._E = torch.tensor(E, dtype=torch.float64)
        self._lower_M = torch.tensor(lower_M, dtype=torch.float64)
        self._l = torch.tensor(lower_c, dtype=torch.float64)
        self._upper_M = torch.tensor(upper_M, dtype=torch.float64)
        self._u = torch.tensor(upper_c, dtype=torch.float64)

        self._objective_bridge = _JaxObjectiveBridge(model, param_name=param_name)

        self._xdim = int(X.shape[1])
        self._ydim = int(model.var_spec.total_size)
        self._num = int(X.shape[0])
        self._neq = int(A.shape[0])
        self._nineq = int(C.shape[0] + 2 * self._ydim)
        self._nknowns = 0
        self._train_frac = float(train_frac)
        self._valid_frac = float(valid_frac)
        self._test_frac = float(test_frac)
        self._device = None
        self._corr_grad_clip = 1e10

        use_completion = self._neq > 0 and self._ydim > self._neq
        self._use_completion = use_completion
        if use_completion:
            if hasattr(module, "dc3_completion_indices"):
                partial, other = module.dc3_completion_indices()
            else:
                from scripts.testcase.nlp_run_dc3 import _choose_completion_indices

                partial, other = _choose_completion_indices(A, seed=split_seed + 17)

            self._partial_vars = np.asarray(partial, dtype=np.int64)
            self._other_vars = np.asarray(other, dtype=np.int64)
            self._A_other_inv = torch.inverse(self._A[:, self._other_vars])
            self._A_partial = self._A[:, self._partial_vars]
            self._d_other_d_partial = -(self._A_other_inv @ self._A_partial)
            self._dy_dz = torch.zeros((self._ydim, self._partial_vars.shape[0]), dtype=torch.float64)
            self._dy_dz[self._partial_vars, :] = torch.eye(self._partial_vars.shape[0], dtype=torch.float64)
            self._dy_dz[self._other_vars, :] = self._d_other_d_partial
        else:
            self._partial_vars = np.arange(self._ydim, dtype=np.int64)
            self._other_vars = np.zeros((0,), dtype=np.int64)
            self._A_other_inv = torch.zeros((0, 0), dtype=torch.float64)
            self._A_partial = torch.zeros((self._neq, self._ydim), dtype=torch.float64)
            self._d_other_d_partial = torch.zeros((0, self._ydim), dtype=torch.float64)
            self._dy_dz = torch.eye(self._ydim, dtype=torch.float64)

    def configure_stabilization(self, *, corr_grad_clip: float) -> None:
        self._corr_grad_clip = float(corr_grad_clip)

    @staticmethod
    def _finite_or_zero(tensor: torch.Tensor) -> torch.Tensor:
        return torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)

    def _clip_tensor(self, tensor: torch.Tensor, clip_abs: float | None = None) -> torch.Tensor:
        tensor = self._finite_or_zero(tensor)
        if clip_abs is None or clip_abs <= 0.0:
            return tensor
        return torch.clamp(tensor, min=-float(clip_abs), max=float(clip_abs))

    def __str__(self):
        return f"CustomDC3-general-{self.ydim}-{self.nineq}-{self.neq}-{self.num}"

    def to(self, device):
        for name, value in list(self.__dict__.items()):
            if torch.is_tensor(value):
                setattr(self, name, value.to(device))
        self._device = device
        return self

    @property
    def xdim(self):
        return self._xdim

    @property
    def ydim(self):
        return self._ydim

    @property
    def num(self):
        return self._num

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
    def train_frac(self):
        return self._train_frac

    @property
    def valid_frac(self):
        return self._valid_frac

    @property
    def test_frac(self):
        return self._test_frac

    @property
    def trainX(self):
        return self._X[: int(self.num * self.train_frac)]

    @property
    def validX(self):
        start = int(self.num * self.train_frac)
        end = int(self.num * (self.train_frac + self.valid_frac))
        return self._X[start:end]

    @property
    def testX(self):
        start = int(self.num * (self.train_frac + self.valid_frac))
        return self._X[start:]

    @property
    def device(self):
        return self._device

    def _eq_rhs(self, X):
        if self._neq == 0:
            return torch.zeros((X.shape[0], 0), device=X.device, dtype=X.dtype)
        return self._b.unsqueeze(0) + X @ self._B.T

    def _ineq_rhs(self, X):
        if self._C.shape[0] == 0:
            return torch.zeros((X.shape[0], 0), device=X.device, dtype=X.dtype)
        return self._d.unsqueeze(0) + X @ self._E.T

    def _bounds(self, X):
        lower = self._l.unsqueeze(0) + X @ self._lower_M.T
        upper = self._u.unsqueeze(0) + X @ self._upper_M.T
        return lower, upper

    def _squash_to_box(self, lower, upper, raw):
        span = torch.clamp(upper - lower, min=1e-6)
        return lower + 0.5 * span * (torch.tanh(raw) + 1.0)

    def obj_fn(self, X, Y):
        return _TorchJaxObjective.apply(X, Y, self._objective_bridge)

    def eq_resid(self, X, Y):
        if self._neq == 0:
            return torch.zeros((X.shape[0], 0), device=Y.device, dtype=Y.dtype)
        return self._finite_or_zero(Y @ self._A.T - self._eq_rhs(X))

    def _affine_ineq_resid(self, X, Y):
        if self._C.shape[0] == 0:
            return torch.zeros((X.shape[0], 0), device=Y.device, dtype=Y.dtype)
        return self._finite_or_zero(Y @ self._C.T - self._ineq_rhs(X))

    def ineq_resid(self, X, Y):
        lower, upper = self._bounds(X)
        lower_resid = lower - Y
        upper_resid = Y - upper
        return self._finite_or_zero(torch.cat([self._affine_ineq_resid(X, Y), lower_resid, upper_resid], dim=1))

    def ineq_dist(self, X, Y):
        return self._finite_or_zero(torch.clamp(self.ineq_resid(X, Y), min=0.0))

    def eq_grad(self, X, Y):
        if self._neq == 0:
            return torch.zeros_like(Y)
        return self._clip_tensor(2.0 * (self.eq_resid(X, Y) @ self._A), self._corr_grad_clip)

    def ineq_grad(self, X, Y):
        grad = torch.zeros_like(Y)

        if self._C.shape[0] > 0:
            aff_dist = torch.clamp(self._affine_ineq_resid(X, Y), min=0.0)
            grad = grad + 2.0 * (aff_dist @ self._C)

        lower, upper = self._bounds(X)
        lower_dist = torch.clamp(lower - Y, min=0.0)
        upper_dist = torch.clamp(Y - upper, min=0.0)
        grad = grad - 2.0 * lower_dist + 2.0 * upper_dist
        return self._clip_tensor(grad, self._corr_grad_clip)

    def ineq_partial_grad(self, X, Y):
        if not self._use_completion:
            return self.ineq_grad(X, Y)
        grad_full = self.ineq_grad(X, Y)
        grad_partial = grad_full @ self._dy_dz
        return self._clip_tensor(grad_partial @ self._dy_dz.T, self._corr_grad_clip)

    def process_output(self, X, Y):
        lower, upper = self._bounds(X)
        return self._finite_or_zero(self._squash_to_box(lower, upper, Y))

    def complete_partial(self, X, Z):
        if not self._use_completion:
            return self.process_output(X, Z)
        rhs = self._eq_rhs(X)
        lower, upper = self._bounds(X)
        z_scaled = self._squash_to_box(lower[:, self._partial_vars], upper[:, self._partial_vars], Z)
        Y = torch.zeros((X.shape[0], self.ydim), device=self.device, dtype=X.dtype)
        Y[:, self._partial_vars] = z_scaled
        Y[:, self._other_vars] = rhs @ self._A_other_inv.T + z_scaled @ self._d_other_d_partial.T
        return self._finite_or_zero(Y)


def _total_loss(data, X, Y, args):
    obj_cost = data.obj_fn(X, Y)
    ineq_dist = data.ineq_dist(X, Y)
    ineq_cost = torch.norm(ineq_dist, dim=1)
    eq_cost = torch.norm(data.eq_resid(X, Y), dim=1)
    return obj_cost + args["softWeight"] * (1 - args["softWeightEqFrac"]) * ineq_cost + args["softWeight"] * args["softWeightEqFrac"] * eq_cost


def _eval_net_with_timing(dc3_method, data, X, solver_net, args, prefix: str, stats: dict, phase_totals=None) -> None:
    eps_converge = args["corrEps"]
    make_prefix = lambda suffix: f"{prefix}_{suffix}"

    start_time = time.time()
    Y = solver_net(X)
    forward_end_time = time.time()

    Ycorr, steps = dc3_method.grad_steps_all(data, X, Y, args)
    corrected_end_time = time.time()

    Ynew = dc3_method.grad_steps(data, X, Y, args)
    raw_end_time = time.time()

    if phase_totals is not None:
        phase_totals["backbone"] += forward_end_time - start_time
        phase_totals["projection"] += raw_end_time - forward_end_time

    dc3_method.dict_agg(stats, make_prefix("time"), corrected_end_time - start_time, op="sum")
    dc3_method.dict_agg(stats, make_prefix("steps"), np.array([steps]))
    dc3_method.dict_agg(stats, make_prefix("loss"), _total_loss(data, X, Ynew, args).detach().cpu().numpy())
    dc3_method.dict_agg(stats, make_prefix("eval"), data.obj_fn(X, Ycorr).detach().cpu().numpy())
    dc3_method.dict_agg(stats, make_prefix("dist"), torch.norm(Ycorr - Y, dim=1).detach().cpu().numpy())
    dc3_method.dict_agg(stats, make_prefix("ineq_max"), torch.max(data.ineq_dist(X, Ycorr), dim=1)[0].detach().cpu().numpy())
    dc3_method.dict_agg(stats, make_prefix("ineq_mean"), torch.mean(data.ineq_dist(X, Ycorr), dim=1).detach().cpu().numpy())
    dc3_method.dict_agg(stats, make_prefix("ineq_num_viol_0"), torch.sum(data.ineq_dist(X, Ycorr) > eps_converge, dim=1).detach().cpu().numpy())
    dc3_method.dict_agg(stats, make_prefix("ineq_num_viol_1"), torch.sum(data.ineq_dist(X, Ycorr) > 10 * eps_converge, dim=1).detach().cpu().numpy())
    dc3_method.dict_agg(stats, make_prefix("ineq_num_viol_2"), torch.sum(data.ineq_dist(X, Ycorr) > 100 * eps_converge, dim=1).detach().cpu().numpy())
    dc3_method.dict_agg(stats, make_prefix("eq_max"), torch.max(torch.abs(data.eq_resid(X, Ycorr)), dim=1)[0].detach().cpu().numpy())
    dc3_method.dict_agg(stats, make_prefix("eq_mean"), torch.mean(torch.abs(data.eq_resid(X, Ycorr)), dim=1).detach().cpu().numpy())
    dc3_method.dict_agg(stats, make_prefix("eq_num_viol_0"), torch.sum(torch.abs(data.eq_resid(X, Ycorr)) > eps_converge, dim=1).detach().cpu().numpy())
    dc3_method.dict_agg(stats, make_prefix("eq_num_viol_1"), torch.sum(torch.abs(data.eq_resid(X, Ycorr)) > 10 * eps_converge, dim=1).detach().cpu().numpy())
    dc3_method.dict_agg(stats, make_prefix("eq_num_viol_2"), torch.sum(torch.abs(data.eq_resid(X, Ycorr)) > 100 * eps_converge, dim=1).detach().cpu().numpy())
    dc3_method.dict_agg(stats, make_prefix("raw_time"), (raw_end_time - corrected_end_time) + (forward_end_time - start_time), op="sum")
    dc3_method.dict_agg(stats, make_prefix("raw_eval"), data.obj_fn(X, Ynew).detach().cpu().numpy())
    dc3_method.dict_agg(stats, make_prefix("raw_ineq_max"), torch.max(data.ineq_dist(X, Ynew), dim=1)[0].detach().cpu().numpy())
    dc3_method.dict_agg(stats, make_prefix("raw_ineq_mean"), torch.mean(data.ineq_dist(X, Ynew), dim=1).detach().cpu().numpy())
    dc3_method.dict_agg(stats, make_prefix("raw_ineq_num_viol_0"), torch.sum(data.ineq_dist(X, Ynew) > eps_converge, dim=1).detach().cpu().numpy())
    dc3_method.dict_agg(stats, make_prefix("raw_ineq_num_viol_1"), torch.sum(data.ineq_dist(X, Ynew) > 10 * eps_converge, dim=1).detach().cpu().numpy())
    dc3_method.dict_agg(stats, make_prefix("raw_ineq_num_viol_2"), torch.sum(data.ineq_dist(X, Ynew) > 100 * eps_converge, dim=1).detach().cpu().numpy())
    dc3_method.dict_agg(stats, make_prefix("raw_eq_max"), torch.max(torch.abs(data.eq_resid(X, Ynew)), dim=1)[0].detach().cpu().numpy())
    dc3_method.dict_agg(stats, make_prefix("raw_eq_mean"), torch.mean(torch.abs(data.eq_resid(X, Ynew)), dim=1).detach().cpu().numpy())
    dc3_method.dict_agg(stats, make_prefix("raw_eq_num_viol_0"), torch.sum(torch.abs(data.eq_resid(X, Ynew)) > eps_converge, dim=1).detach().cpu().numpy())
    dc3_method.dict_agg(stats, make_prefix("raw_eq_num_viol_1"), torch.sum(torch.abs(data.eq_resid(X, Ynew)) > 10 * eps_converge, dim=1).detach().cpu().numpy())
    dc3_method.dict_agg(stats, make_prefix("raw_eq_num_viol_2"), torch.sum(torch.abs(data.eq_resid(X, Ynew)) > 100 * eps_converge, dim=1).detach().cpu().numpy())


def _train_net_with_logging(dc3_method, data, args: dict, save_dir: str):
    from torch.utils.data import DataLoader, TensorDataset
    import torch.optim as optim

    solver_step = args["lr"]
    nepochs = int(args["epochs"])
    batch_size = int(args["batchSize"])

    train_dataset = TensorDataset(data.trainX)
    valid_dataset = TensorDataset(data.validX)
    test_dataset = TensorDataset(data.testX)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    train_eval_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    solver_net = dc3_method.NNSolver(data, args)
    solver_net.to(dc3_method.DEVICE)
    solver_opt = optim.Adam(solver_net.parameters(), lr=solver_step)

    stats = {}
    train_epoch_time_total = 0.0
    val_epoch_time_total = 0.0
    train_epoch_time_tracked = 0.0
    val_epoch_time_tracked = 0.0
    backbone_total_sec = 0.0
    projection_total_sec = 0.0
    backward_total_sec = 0.0
    optimizer_total_sec = 0.0
    backbone_tracked_total_sec = 0.0
    projection_tracked_total_sec = 0.0
    backward_tracked_total_sec = 0.0
    optimizer_tracked_total_sec = 0.0
    training_wall_t0 = time.time()
    for ep in range(nepochs):
        epoch_stats = {}
        val_phase_totals = {"backbone": 0.0, "projection": 0.0}

        solver_net.train()
        train_epoch_t0 = time.time()
        for Xtrain in train_loader:
            Xtrain = Xtrain[0].to(dc3_method.DEVICE)
            start_time = time.time()
            solver_opt.zero_grad()
            Yhat_train = solver_net(Xtrain)
            after_forward = time.time()
            Ynew_train = dc3_method.grad_steps(data, Xtrain, Yhat_train, args)
            after_projection = time.time()
            train_loss = _total_loss(data, Xtrain, Ynew_train, args)
            after_loss = time.time()
            train_loss.sum().backward()
            after_backward = time.time()
            torch.nn.utils.clip_grad_norm_(solver_net.parameters(), max_norm=float(args.get("trainGradClipNorm", 1e3)))
            solver_opt.step()
            after_optimizer = time.time()
            train_time = after_optimizer - start_time
            backbone_total_sec += after_forward - start_time
            projection_total_sec += after_loss - after_forward
            backward_total_sec += after_backward - after_loss
            optimizer_total_sec += after_optimizer - after_backward
            if should_track_epoch(ep, nepochs):
                backbone_tracked_total_sec += after_forward - start_time
                projection_tracked_total_sec += after_loss - after_forward
                backward_tracked_total_sec += after_backward - after_loss
                optimizer_tracked_total_sec += after_optimizer - after_backward
            dc3_method.dict_agg(epoch_stats, "train_loss", train_loss.detach().cpu().numpy())
            dc3_method.dict_agg(epoch_stats, "train_time", train_time, op="sum")
        train_epoch_elapsed = time.time() - train_epoch_t0
        train_epoch_time_total += train_epoch_elapsed
        if should_track_epoch(ep, nepochs):
            train_epoch_time_tracked += train_epoch_elapsed

        solver_net.eval()
        with torch.no_grad():
            val_epoch_t0 = time.time()
            for Xbatch in valid_loader:
                Xbatch = Xbatch[0].to(dc3_method.DEVICE)
                _eval_net_with_timing(dc3_method, data, Xbatch, solver_net, args, "valid", epoch_stats, val_phase_totals)
            val_epoch_elapsed = time.time() - val_epoch_t0
            val_epoch_time_total += val_epoch_elapsed
            backbone_total_sec += val_phase_totals["backbone"]
            projection_total_sec += val_phase_totals["projection"]
            if should_track_epoch(ep, nepochs):
                val_epoch_time_tracked += val_epoch_elapsed
                backbone_tracked_total_sec += val_phase_totals["backbone"]
                projection_tracked_total_sec += val_phase_totals["projection"]

            for Xbatch in train_eval_loader:
                Xbatch = Xbatch[0].to(dc3_method.DEVICE)
                _eval_net_with_timing(dc3_method, data, Xbatch, solver_net, args, "train", epoch_stats)
            for Xbatch in test_loader:
                Xbatch = Xbatch[0].to(dc3_method.DEVICE)
                _eval_net_with_timing(dc3_method, data, Xbatch, solver_net, args, "test", epoch_stats)

        if (ep % int(args.get("printEvery", 10))) == 0 or ep == nepochs - 1:
            print(
                f"ep {ep:05d} | "
                f"train loss {_mean_stat(epoch_stats, 'train_loss'):.6e} "
                f"obj {_mean_stat(epoch_stats, 'train_eval'):.6e} "
                f"cons {_dc3_consistency(epoch_stats, 'train'):.6e} "
                f"viol {_dc3_violation(epoch_stats, 'train'):.6e} || "
                f"val loss {_mean_stat(epoch_stats, 'valid_loss'):.6e} "
                f"obj {_mean_stat(epoch_stats, 'valid_eval'):.6e} "
                f"cons {_dc3_consistency(epoch_stats, 'valid'):.6e} "
                f"viol {_dc3_violation(epoch_stats, 'valid'):.6e}"
            )

        if args["saveAllStats"]:
            if ep == 0:
                for key, value in epoch_stats.items():
                    stats[key] = np.expand_dims(np.array(value), axis=0)
            else:
                for key, value in epoch_stats.items():
                    stats[key] = np.concatenate((stats[key], np.expand_dims(np.array(value), axis=0)))
        else:
            stats = epoch_stats

        if (ep % int(args["resultsSaveFreq"])) == 0:
            _save_dc3_training_artifacts(stats, solver_net, save_dir)

    training_wall_time_sec = time.time() - training_wall_t0
    _save_dc3_training_artifacts(stats, solver_net, save_dir)
    profile = {
        "training_wall_time_sec": training_wall_time_sec,
        "train_epoch_time_total_sec": train_epoch_time_total,
        "val_epoch_time_total_sec": val_epoch_time_total,
        "train_epoch_time_tracked_sec": train_epoch_time_tracked,
        "val_epoch_time_tracked_sec": val_epoch_time_tracked,
        "backbone_total_sec": backbone_total_sec,
        "projection_total_sec": projection_total_sec,
        "backward_total_sec": backward_total_sec,
        "optimizer_total_sec": optimizer_total_sec,
        "backbone_tracked_total_sec": backbone_tracked_total_sec,
        "projection_tracked_total_sec": projection_tracked_total_sec,
        "backward_tracked_total_sec": backward_tracked_total_sec,
        "optimizer_tracked_total_sec": optimizer_tracked_total_sec,
        "epochs": nepochs,
        "train_batches_per_epoch": len(train_loader),
        "val_batches_per_epoch": len(valid_loader),
    }
    return solver_net, stats, profile


def _print_final_summary(dc3_method, data, solver_net, args: dict, profile: dict) -> dict:
    from torch.utils.data import DataLoader, TensorDataset

    batch_size = int(args["batchSize"])
    eval_sets = (
        TensorDataset(data.trainX),
        TensorDataset(data.validX),
    )

    eq_max = 0.0
    ineq_max = 0.0
    bound_max = 0.0
    objective_total = 0.0
    objective_count = 0

    solver_net.eval()
    with torch.no_grad():
        for dataset in eval_sets:
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            for (Xbatch,) in loader:
                Xbatch = Xbatch.to(dc3_method.DEVICE)
                Ypred = solver_net(Xbatch)
                Ycorr, _ = dc3_method.grad_steps_all(data, Xbatch, Ypred, args)

                if data.neq > 0:
                    eq_max = max(eq_max, float(torch.max(torch.abs(data.eq_resid(Xbatch, Ycorr))).item()))
                ineq_max = max(ineq_max, float(torch.max(data.ineq_dist(Xbatch, Ycorr)).item()))
                lower, upper = data._bounds(Xbatch)
                bound_violation = torch.maximum(torch.clamp(lower - Ycorr, min=0.0), torch.clamp(Ycorr - upper, min=0.0))
                if bound_violation.numel() > 0:
                    bound_max = max(bound_max, float(torch.max(bound_violation).item()))

                objective_values = data.obj_fn(Xbatch, Ycorr)
                objective_total += float(torch.sum(objective_values).item())
                objective_count += int(objective_values.numel())

    objective_mean = objective_total / max(1, objective_count)
    timing_summary = summarize_timing_profile(profile)
    profiled_total = (
        float(timing_summary["backbone_total_sec"])
        + float(timing_summary["projection_total_sec"])
        + float(timing_summary["backward_total_sec"])
        + float(timing_summary["optimizer_total_sec"])
    )

    print("\n=== Constraint violation (max over train+val) ===")
    print(f"Equality   ||A y - (b+Bx)||_inf : {eq_max:.6e}")
    print(f"Inequality max(·,0)_inf         : {ineq_max:.6e}")
    print(f"Bounds     max(lb,ub)_inf       : {bound_max:.6e}\n")

    print("=== Training evaluation ===")
    print(f"Projected objective: {objective_mean:.6e}\n")

    print("=== Profiled training time distribution ===")
    print(f"Training wall time: {float(profile['training_wall_time_sec']):.3f}s")
    print(
        f"Average epoch time ({timing_window_label(int(profile['epochs']))}): "
        f"train {float(timing_summary['avg_train_epoch_time_sec']):.3f}s "
        f"val {float(timing_summary['avg_val_epoch_time_sec']):.3f}s "
        f"total {float(timing_summary['avg_total_epoch_time_sec']):.3f}s"
    )
    print(
        f"Average batch time ({timing_window_label(int(profile['epochs']))}): "
        f"train {float(timing_summary['avg_train_batch_time_sec']):.4f}s "
        f"val {float(timing_summary['avg_val_batch_time_sec']):.4f}s "
        f"overall {float(timing_summary['avg_total_batch_time_sec']):.4f}s"
    )
    if profiled_total > 0.0:
        print(f"Backbone forward : {float(timing_summary['time_backbone_percent']):6.2f}% (train + val)")
        print(f"Projection       : {float(timing_summary['time_projection_percent']):6.2f}% (train + val)")
        print(f"Backward         : {float(timing_summary['time_backward_percent']):6.2f}% (train only)")
        print(f"Optimizer update : {float(timing_summary['time_optimizer_percent']):6.2f}% (train only)")
    print("")

    summary = {
        "eq_inf": eq_max,
        "ineq_inf": ineq_max,
        "bound_inf": bound_max,
        "projected_objective": objective_mean,
        "training_wall_time_sec": float(profile["training_wall_time_sec"]),
    }
    summary.update(timing_summary)
    return summary


def _build_history_payload(stats: dict) -> dict:
    epochs = list(range(len(unified._epoch_mean_series(stats, "train_eval"))))
    train_violation = np.maximum(
        np.asarray(unified._epoch_max_series(stats, "train_ineq_max"), dtype=float),
        np.asarray(unified._epoch_max_series(stats, "train_eq_max"), dtype=float),
    ).tolist()
    val_violation = np.maximum(
        np.asarray(unified._epoch_max_series(stats, "valid_ineq_max"), dtype=float),
        np.asarray(unified._epoch_max_series(stats, "valid_eq_max"), dtype=float),
    ).tolist()
    return {
        "epochs": epochs,
        "train_objective": unified._epoch_mean_series(stats, "train_eval"),
        "val_objective": unified._epoch_mean_series(stats, "valid_eval"),
        "train_violation": [float(v) for v in train_violation],
        "val_violation": [float(v) for v in val_violation],
        "framework": "dc3",
    }


def _summary_metrics_from_stats(stats: dict, problem) -> dict:
    train_count = int(problem.trainX.shape[0])
    val_count = int(problem.validX.shape[0])
    return {
        "max_equality": float(unified._combined_final_max(stats, "train_eq_max", "valid_eq_max")),
        "mean_equality": float(
            unified._weighted_final_mean(
                stats,
                "train_eq_mean",
                "valid_eq_mean",
                train_count=train_count,
                val_count=val_count,
            )
        ),
        "max_inequality": float(unified._combined_final_max(stats, "train_ineq_max", "valid_ineq_max")),
        "mean_inequality": float(
            unified._weighted_final_mean(
                stats,
                "train_ineq_mean",
                "valid_ineq_mean",
                train_count=train_count,
                val_count=val_count,
            )
        ),
        "consistency": float(
            unified._weighted_final_mean(
                stats,
                "train_dist",
                "valid_dist",
                train_count=train_count,
                val_count=val_count,
            )
        ),
    }


def _run_single_case(
    case_dir: Path,
    data_cfg: dict,
    cfg_dict: dict,
    proj_cfg: dict,
    model_def_path: Path,
    *,
    output_dir_override: Path | None = None,
) -> unified.RunArtifacts:
    if torch is None:
        raise RuntimeError("DC3 runner requires `torch`. Install PyTorch before running this script.")

    seed_torch_runtime(int(cfg_dict.get("seed", 42)))
    generator = build_problem_generator(data_cfg, case_dir=case_dir)
    model = build_problem_model(dtype=jnp.float64, case_dir=case_dir)
    module = load_model_module(case_dir)
    dataset = general_local.ensure_cached_dataset(
        case_dir,
        data_cfg,
        model_def_path,
        lambda: general_local._generate_dataset(generator, data_cfg),
        force=bool(data_cfg.get("force_regenerate", False)),
    )

    train_frac, valid_frac, test_frac = _resolve_split_fracs(cfg_dict)
    problem = GeneralDC3Problem(
        model=model,
        param_name=getattr(module, "PARAM_NAME", "x"),
        module=module,
        X=dataset.X,
        train_frac=train_frac,
        valid_frac=valid_frac,
        test_frac=test_frac,
        split_seed=int(cfg_dict.get("seed", 42)),
    )

    dc3_method = _load_dc3_method_module()
    args = _build_dc3_args(cfg_dict, use_completion=problem._use_completion)
    save_dir = Path(output_dir_override) if output_dir_override is not None else _save_dir(dataset.dataset_dir, data_cfg, cfg_dict, args)
    save_dir.mkdir(parents=True, exist_ok=True)

    problem.to(dc3_method.DEVICE)
    problem.configure_stabilization(corr_grad_clip=float(args.get("corrGradClip", 1e10)))
    general_local._write_local_run_configs(
        save_dir,
        general_local._normalize_local_data_cfg(data_cfg, model_def_path),
        cfg_dict,
        proj_cfg,
        model_def_path,
    )
    with open(save_dir / "args.json", "w", encoding="utf-8") as fh:
        json.dump(args, fh, indent=2, sort_keys=True)
    with open(save_dir / "args.dict", "wb") as fh:
        pickle.dump(args, fh)

    print(f"Dataset: {dataset.dataset_dir}")
    print(f"DC3 save dir: {save_dir}")
    print("Problem type: general")
    print(f"Device: {dc3_method.DEVICE}")
    print(f"Dimensions: n_x={problem.xdim} n_y={problem.ydim} n_eq={problem.neq} n_ineq={problem.nineq}")
    print(f"Completion: {'enabled' if problem._use_completion else 'disabled'}")
    print(f"Splits: train={problem.trainX.shape[0]}  valid={problem.validX.shape[0]}  test={problem.testX.shape[0]}")

    solver_net, stats, profile = _train_net_with_logging(dc3_method, problem, args, str(save_dir))
    final_metrics = _print_final_summary(dc3_method, problem, solver_net, args, profile)
    summary = _summarize_stats(stats)
    summary.update(final_metrics)
    summary["dataset_dir"] = str(dataset.dataset_dir)
    summary["save_dir"] = str(save_dir)
    summary["framework"] = "dc3"
    summary["objective_value"] = float(summary["projected_objective"])
    summary["optimality_gap"] = None
    summary["relative_objective_gap"] = None
    summary.update(_summary_metrics_from_stats(stats, problem))

    history_payload = _build_history_payload(stats)
    history_path = save_dir / "run_history.json"
    plot_path = save_dir / "compare_metrics.png"
    metrics_path = save_dir / "summary.json"
    solver_state_path = save_dir / "solver_net.dict"
    stats_path = save_dir / "stats.dict"

    _write_json(history_path, history_payload)
    save_objective_value_violation_plot(
        plot_path,
        epochs=history_payload["epochs"],
        train_objective=history_payload["train_objective"],
        val_objective=history_payload["val_objective"],
        train_violation=history_payload["train_violation"],
        val_violation=history_payload["val_violation"],
        title="General Training",
        series_label="DC3",
    )
    summary["space_mb"] = general_local._artifact_size_mb(
        history_path,
        plot_path,
        solver_state_path,
        stats_path,
        save_dir / "args.json",
        save_dir / "args.dict",
    )
    _write_json(metrics_path, summary)

    print(f"[dc3] Saved: {save_dir}")
    return unified.RunArtifacts(
        framework="dc3",
        dataset_dir=dataset.dataset_dir,
        run_dir=save_dir,
        history_path=history_path,
        metrics_path=metrics_path,
        plot_path=plot_path,
    )


def _run_multi_seed(
    case_dir: Path,
    data_cfg: dict,
    cfg_dict: dict,
    proj_cfg: dict,
    model_def_path: Path,
) -> int:
    framework = "dc3"
    base_seed = int(cfg_dict.get("seed", 42))
    num_seeds = int(cfg_dict.get("num_seeds", 10))
    if num_seeds <= 0:
        raise ValueError("num_seeds must be positive when run_multiple_seed is true.")
    seeds = [base_seed + idx for idx in range(num_seeds)]

    dataset_root = general_local.dataset_dir(case_dir, data_cfg, model_def_path)
    multi_dir = unified._framework_multi_dir(dataset_root, framework)
    multi_dir.mkdir(parents=True, exist_ok=True)
    general_local._write_local_run_configs(
        multi_dir,
        general_local._normalize_local_data_cfg(data_cfg, model_def_path),
        cfg_dict,
        proj_cfg,
        model_def_path,
    )

    run_manifests = []
    history_payloads = []
    for seed in seeds:
        run_cfg = copy.deepcopy(cfg_dict)
        run_cfg["seed"] = int(seed)
        run_cfg["run_multiple_seed"] = False
        print("")
        print(f"[multi-seed] Running dc3 with config.seed={seed}")
        seed_dir = unified._framework_seed_dir(dataset_root, framework, int(seed))
        artifacts = _run_single_case(
            case_dir,
            data_cfg,
            run_cfg,
            proj_cfg,
            model_def_path,
            output_dir_override=seed_dir,
        )
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
        save_shadow_objective_value_violation_plot(
            shadow_plot_path,
            epochs=epochs,
            train_objective_runs=[payload["train_objective"] for payload in history_payloads],
            val_objective_runs=[payload["val_objective"] for payload in history_payloads],
            train_violation_runs=[payload["train_violation"] for payload in history_payloads],
            val_violation_runs=[payload["val_violation"] for payload in history_payloads],
            series_label="DC3",
        )

    summary_path = multi_dir / "multi_seed_summary.json"
    _write_json(
        summary_path,
        {
            "framework": framework,
            "framework_label": "DC3",
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
        data_cfg=general_local._normalize_local_data_cfg(data_cfg, model_def_path),
        cfg_dict=cfg_dict,
        proj_cfg=proj_cfg,
        framework=framework,
        seeds=seeds,
        extra={"summary_path": str(summary_path)},
    )
    print(f"[multi-seed] Saved summary: {summary_path}")
    print(f"[multi-seed] Updated family metadata: {metadata_path}")
    return 0


def run_case(case_dir: Path | None = None, _path_arg: str | None = None) -> int:
    case_dir = default_case_dir() if case_dir is None else Path(case_dir)
    data_cfg = _load_json(case_dir / "data.json")
    cfg_dict = _load_json(case_dir / "config.json")
    proj_cfg_path = case_dir / "proj.json"
    proj_cfg = _load_json(proj_cfg_path) if proj_cfg_path.exists() else {}
    model_path = model_definition_path(case_dir)
    _run_single_case(case_dir, data_cfg, cfg_dict, proj_cfg, model_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(run_case())
