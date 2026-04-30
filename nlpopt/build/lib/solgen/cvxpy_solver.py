from __future__ import annotations

from typing import Optional

import cvxpy as cp
import jax.numpy as jnp
import numpy as np

from jaxmodel import JaxNLPModel

from .types import QuadraticProgramData


def _as_array(value, size: int, fill: float) -> np.ndarray:
    if value is None:
        return np.full((size,), fill, dtype=np.float64)
    return np.asarray(value, dtype=np.float64).reshape((size,))


def solve_quadratic_program(
    problem: QuadraticProgramData,
    *,
    solver: Optional[str] = None,
    warm_start: Optional[np.ndarray] = None,
    verbose: bool = False,
) -> tuple[str, np.ndarray, float]:
    Q = np.asarray(problem.Q, dtype=np.float64)
    c = np.asarray(problem.c, dtype=np.float64).reshape((-1,))
    A = np.asarray(problem.A, dtype=np.float64)
    b = np.asarray(problem.b, dtype=np.float64).reshape((-1,))
    C = np.asarray(problem.C, dtype=np.float64)
    d = np.asarray(problem.d, dtype=np.float64).reshape((-1,))

    n = c.shape[0]
    l = _as_array(problem.l, n, -np.inf)
    u = _as_array(problem.u, n, np.inf)

    y = cp.Variable(n)
    constraints = []
    if A.size != 0:
        constraints.append(A @ y == b)
    if C.size != 0:
        constraints.append(C @ y <= d)
    constraints.append(y >= l)
    constraints.append(y <= u)

    Qsym = 0.5 * (Q + Q.T)
    objective = cp.Minimize(0.5 * cp.quad_form(y, Qsym) + c @ y)
    program = cp.Problem(objective, constraints)

    if warm_start is not None:
        y.value = np.asarray(warm_start, dtype=np.float64).reshape((n,))

    chosen_solver = solver or ("OSQP" if np.allclose(Qsym, np.diag(np.diag(Qsym))) else "SCS")
    try:
        program.solve(solver=chosen_solver, warm_start=True, verbose=verbose)
    except cp.SolverError:
        fallback = "SCS" if chosen_solver != "SCS" else None
        if fallback is None:
            raise
        program.solve(solver=fallback, warm_start=True, verbose=verbose)

    if y.value is None:
        raise RuntimeError(f"CVXPY failed with status '{program.status}'")

    return str(program.status), np.asarray(y.value, dtype=np.float64).reshape((n,)), float(program.value)


def solve_jaxmodel_direct(
    model: JaxNLPModel,
    params,
    *,
    solver: Optional[str] = None,
    warm_start: Optional[np.ndarray] = None,
    verbose: bool = False,
) -> tuple[str, np.ndarray, float]:
    n = model.var_spec.total_size
    y = cp.Variable(n)

    y0 = jnp.zeros((n,), dtype=model.dtype)
    Q = np.asarray(model.hess_y_objective(params, y0), dtype=np.float64)
    c = np.asarray(model.grad_y_objective(params, y0), dtype=np.float64).reshape((n,))
    objective = cp.Minimize(0.5 * cp.quad_form(y, 0.5 * (Q + Q.T)) + c @ y)

    constraints = []
    for entry in model.constraints:
        meta = entry.metadata or {}
        if entry.structure == "affine" and meta.get("type") == "affine_block":
            J = np.asarray(meta["jac_matrix"], dtype=np.float64)
            rhs = np.asarray(meta["rhs_const"], dtype=np.float64).reshape((-1,))
            for mat, name in meta.get("x_blocks", []):
                rhs = rhs + np.asarray(mat, dtype=np.float64) @ np.ravel(np.asarray(params[name], dtype=np.float64))
            expr = J @ y
            constraints.append(expr == rhs if entry.kind == "eq" else expr <= rhs)
            continue

        if entry.structure == "quadratic" and meta.get("type") == "quadratic_scalar":
            Qc = np.asarray(meta["Q"], dtype=np.float64)
            cc = np.asarray(meta["c"], dtype=np.float64).reshape((n,))
            rhs = float(np.asarray(meta["rhs_const"], dtype=np.float64))
            if meta.get("x_coeff") is not None:
                rhs = rhs + float(np.asarray(meta["x_coeff"], dtype=np.float64) @ np.ravel(np.asarray(params[meta["x_name"]], dtype=np.float64)))
            expr = 0.5 * cp.quad_form(y, 0.5 * (Qc + Qc.T)) + cc @ y
            constraints.append(expr == rhs if entry.kind == "eq" else expr <= rhs)
            continue

        raise NotImplementedError(f"Unsupported direct CVXPY constraint '{entry.name}' with structure '{entry.structure}'.")

    lb = model.lower_bounds(params)
    ub = model.upper_bounds(params)
    if lb is not None:
        constraints.append(y >= np.asarray(lb, dtype=np.float64))
    if ub is not None:
        constraints.append(y <= np.asarray(ub, dtype=np.float64))

    program = cp.Problem(objective, constraints)
    if warm_start is not None:
        y.value = np.asarray(warm_start, dtype=np.float64).reshape((n,))

    chosen_solver = solver or "SCS"
    program.solve(solver=chosen_solver, warm_start=True, verbose=verbose)
    if y.value is None:
        raise RuntimeError(f"CVXPY failed with status '{program.status}'")
    return str(program.status), np.asarray(y.value, dtype=np.float64).reshape((n,)), float(program.value)
