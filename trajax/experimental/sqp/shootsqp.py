# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# pylint: disable=invalid-name
"""Trajectory Optimization using Shooting-based SQP.

Description of algorithm provided here:
https://arxiv.org/pdf/2109.07081.pdf

Four SQP modes: {OPEN, STABLE, SENS, APPROX_SENS}.

OPEN: open-loop (standard) SQP.
STABLE: use TV-LQR generated gains for tracking-based closed-loop rollouts.
SENS: sensitivity-generated gains for closed-loop rollouts (see paper above).
APPROX_SENS: approximate version of SENS using a simplified gain computation.
"""

import collections
import dataclasses
import enum
import functools
import time
from typing import Any, List, Optional, Tuple

from absl import logging
import jax
from jax import lax
from jax import tree_util
import jax.numpy as jnp
from ml_collections import config_dict as configdict
from trajax import optimizers
from trajax.experimental.sqp import ddp
from trajax.experimental.sqp import solver_base
from trajax.experimental.sqp import trajqp as qp
from trajax.experimental.sqp import util


def update_params_default(params: Tuple[Any, Any],
                          U: jnp.ndarray,
                          X: jnp.ndarray) -> Tuple[Any, Any]:
  """Updates parameters for the cost and state constraint functions.

  Default function to override.

  Use with caution: parameters should ideally be held fixed between iterations.

  Args:
    params: current parameter tuples (cost_params,state_constraints_params)
    U: current control trajectory soln to update parameters; (T, m) array
    X: current state trajectory soln to update parameters; (T+1, n) array

  Returns:
    updated_params: updated pair of parameter lists
  """
  del U, X
  return params


class QP_SOLVER(enum.Enum):
  # First two solvers use cvxpy. The third uses optimizers.constrained_ilqr
  QP_ECOS = "ecos"
  QP_CVX = "cvxopt"
  QP_ALILQR = "alilqr"


class SQP_METHOD(enum.Enum):
  OPEN = 1
  STABLE = 2
  SENS = 3
  APPROX_SENS = 4


class Status(enum.Enum):
  SOLVED = 1
  ERROR = -1
  STALLED = 2
  MAXITER = 3


@dataclasses.dataclass
class SQPSolution:
  """Dataclass for representing output of ShootSQP.solve().

  iterations: total # of iterations
  history: dict of lists: ('steplength', 'obj', 'min_viol'),
            documenting solution progress; each list is of length iterations.
            Two additional lists for SENS:
            {'ddp_err': |du_qp-du_ddp|_2,
            'ddp_err_grad': |ilqr_grad|_inf)}.
  times: (iterations,)-size list of per iteration computation times.
  objective: final solution objective.
  status: exit status, one of Status enums:
    {1: SOLVED - i.e., KKT conditions satisfied to tolerance;
      2: STALLED - i.e., step-size stalled,
                          does not imply KKT satisfaction;
      3: MAXITER - i.e., max SQP iterations completed,
                              does not imply KKT satisfaction;
    -1: ERROR - i.e., QP problem infeasible (if using cvxpy),
                          or other error - debug required.
    }
  primals: tuple of returned solution trajectories;
            (U: [T, m]-ndarray, X: [T+1, n]-ndarray).
  duals: tuple of returned duals solution trajectories;
          (Yu: [T, n_cu]-ndarray, Yx: [T+1, n_cx]-ndarray).
  kkt_residuals: dictionary of kkt residuals.
  """
  iterations: int
  history: Optional[dict[str, List[float]]]
  times: Optional[List[float]]
  objective: float
  status: Status
  primals: Tuple[jnp.ndarray, jnp.ndarray]
  duals: Tuple[jnp.ndarray, jnp.ndarray]
  kkt_residuals: dict[str, Any]


class ShootSQP(solver_base.TrajectoryOptimizationSolver):
  """Shooting SQP solver for Trajectory optimization.

  NOTE: See solver_base.TrajectoryOptimizationSolver for args list,
        and see below for solver user_options.

  Algorithm pseudocode:
  (1) Compute parameters for QP sub-problem;
      (LTV dynamics, linear inequalities, quadratized cost).
  (2) Solve QP sub-problem to compute perturbation control trajectory,
      (and perturbation dual trajectory).
  (3) Rollout perturbed control trajectory through dynamics to update
      primal-dual solution, using augmented Lagrangian as the merit function.
  (4) Check KKT termination conditions.

  Distinction between the SQP variants comes from Step (3), where one can use
  different rollout schemes, e.g., open-loop, stabilized closed-loop, etc. See
  paper linked above for more details.
  """

  def __init__(self,
               n: int,
               m: int,
               T: int,
               dynamics,
               cost,
               control_bounds,
               state_constraint,
               s1_ind: Optional[Tuple[int, ...]] = None,
               update_params=update_params_default,
               **user_options):
    """Initialize SQP solver class."""

    # Initialize the problem.
    super().__init__(n, m, T, dynamics, cost, control_bounds, state_constraint,
                     s1_ind)
    # Setup solver options.
    opt = dict(
        method=SQP_METHOD.OPEN,  # shooting method, see 'SQP_<strings>' above.
        proj_init=False,  # Use projected initialization.
        hess="full",  # Use "full" or "gn" (Gauss-Newton) Hessian.
        mu=1.0,  # Tuning parameter for TV-LQR control cost; range: (0., 1.].
        qp_solver=QP_SOLVER.QP_ECOS,  # QP sub-problem solver.
        qp_options={},  # qp solver options.
        do_log=True,  # whether or not to store solution history log.
        debug=False,  # for printing line-search debug statements.
        verbose=False,  # for printing optimization progress.
        ls_sigma=0.4,  # linesearch parameter; range: (0, ls_eta].
        ls_eta=0.4,  # linesearch parameter; range: [ls_sigma, 0.5).
        ls_beta=0.75,  # backtrack parameter for linesearch; range: (0.5, 1).
        ls_alpha_lb=1e-5,  # lower-bound for step-size; range: (1e-6, 1e-4).
        primal_tol=1e-3,  # primal convergence tolerance; range: (1e-4, 1e-1).
        dual_tol=1e-3,  # dual convergence tolerance; range: (1e-4, 1e-1).
        ddp_options={},  # options for sensitivity gain computation.
        max_iter=100)  # maximum number of SQP iterations.
    opt.update(user_options)
    self.opt = configdict.ConfigDict(opt)

    # Get rollout method.
    rollouts = util.Rollouts(dynamics, self._state_wrap, self._vec_wrap,
                             control_bounds, T)
    self._init_rollout = jax.jit(rollouts.default_rollout)
    if self.opt.method == SQP_METHOD.OPEN:
      self._rollout = rollouts.get_open_rollout()
    elif self.opt.method in (
        SQP_METHOD.STABLE, SQP_METHOD.SENS, SQP_METHOD.APPROX_SENS):
      self._rollout = rollouts.get_closed_rollout()
    else:
      raise NotImplementedError("Method type: "
                                f"{self.opt.method} not implemented.")
    self._proj_rollout = rollouts.get_proj_rollout()
    self.update_params = update_params

    # Setup QP Problem.
    if self.opt.qp_solver == QP_SOLVER.QP_ALILQR:
      self._QP = qp.TrajQPalqr(n, m, T, self._n_cu, self._n_cx,
                               **self.opt.qp_options)
    elif self.opt.qp_solver in (QP_SOLVER.QP_ECOS, QP_SOLVER.QP_CVX):
      self._QP = qp.TrajQPcvx(n, m, T, self._n_cu, self._n_cx,
                              self.opt.qp_solver.value, **self.opt.qp_options)
    else:
      raise NotImplementedError("QP solver: "
                                f"{self.opt.qp_solver} not implemented.")
    self._qp_soln = self._QP.soln

    # Setup sensitivity-based gain computation.
    if self.opt.method == SQP_METHOD.SENS:
      self._DDP = ddp.DDP_QP(n, m, T, self._n_cu, self._n_cx,
                             **self.opt.ddp_options)
    elif self.opt.method == SQP_METHOD.APPROX_SENS:
      self._DDP = ddp.APPROX_DDP_QP(n, m, T, self._n_cu, self._n_cx,
                                    **self.opt.ddp_options)

  def solve(self, x0, U0, X0=None, params=((), ())):
    """Solve the problem for the given initial condition.

    Args:
      x0: initial condition, ndarray: [n,]
      U0: initial guess for control trajectory, ndarray: [T, m]
      X0: initial guess for state trajectory, ndarray: [T+1, n]
      params: tuple of (cost params, state-constraint params).

    Returns:
      SQPSolution instance.

    Raises:
      ValueError: if proj_init=True and X0 is None.
    """

    max_iter = self.opt.max_iter
    verbose = self.opt.verbose
    primal_tol = self.opt.primal_tol
    do_log = self.opt.do_log

    # Initialize X0.
    if self.opt.proj_init:
      # "Projected initialization"
      U0, X0 = self._proj_init(x0, U0, X0, params[0])
    else:
      # U0 only initialization.
      X0 = self._vec_wrap(self._init_rollout(U0, x0))

    # Initialize duals
    YU_0 = jnp.zeros((self._T, self._n_cu))
    YX_0 = jnp.zeros((self._T + 1, self._n_cx))

    # Complete initialization.
    primals = (U0, X0)
    duals = (YU_0, YX_0)
    self.soln = (primals, duals)
    self._qp_soln = self._QP.reset_soln()
    Rho = jnp.zeros(self._T + 1)
    params = self.update_params(params, primals[0], primals[1])
    self.it = 0

    status = None
    if do_log:
      logs = ["steplength", "obj", "min_viol"]
      if self.opt.method in (SQP_METHOD.SENS, SQP_METHOD.APPROX_SENS):
        logs += ["ddp_err", "ddp_err_grad"]
      history = {log: collections.deque(maxlen=max_iter) for log in logs}
      times = collections.deque(maxlen=max_iter)
    else:
      history, times = (None, None)

    # Get initial optimality conditions
    kkt_resid, qp_params = self._get_KKT(primals, duals, params)

    while status is None:

      if verbose:
        print(kkt_resid)
      opt_true = self._check_opt(kkt_resid, primals, duals)

      # Check for termination.
      if opt_true:
        status = Status.SOLVED
        continue

      # Do SQP step.
      time_start = time.time() if do_log else None
      try:
        primals, duals, Rho, alpha, dU, ddp_errs = self._do_step(
            primals, duals, Rho, qp_params, params, self.it)
      except Exception:  # pylint: disable=broad-except
        logging.info("Error - QP problem infeasible or other error!")
        status = Status.ERROR
        continue
      if do_log:
        assert times is not None
        times.append(time.time() - time_start)

      # Update solution.
      self.soln = (primals, duals)
      params = self.update_params(params, primals[0], primals[1])
      kkt_resid, qp_params = self._get_KKT(primals, duals, params)

      # Record iteration summary
      if do_log:
        self._record_summary(history, alpha, kkt_resid, ddp_errs)
      self.it += 1

      step_size = jnp.max(jnp.abs(dU))
      if verbose:
        obj = kkt_resid["obj"]
        print(f"SOLVE: it: {self.it}, obj: {obj}, |dU|: {step_size}, "
              f"step-length: {alpha}, rho: {jnp.max(Rho)}")
      if step_size < primal_tol * (1 + jnp.linalg.norm(primals[0], "fro")):
        status = Status.STALLED
      elif self.it >= max_iter:
        status = Status.MAXITER

    obj = kkt_resid["obj"]
    if verbose:
      print(f"Status: {status.name}, obj: {obj}")

    return SQPSolution(
        self.it, history, times, obj, status, primals, duals, kkt_resid)

  def _proj_init(self, x0, U0, X0, cost_params):
    """Perform projected initialization."""
    mu = self.opt.mu
    if X0 is None:
      raise ValueError("X0 must not be None if using projected init.")
    X0 = X0.at[0].set(x0)
    dyn_params = self._dynamics_jacobians(X0[:-1], U0, self._timesteps[:-1])
    K = self._get_LQR_control_gains(U0, X0, dyn_params, mu, cost_params)
    return self._proj_rollout(U0, X0, K)

  @functools.partial(jax.jit, static_argnums=(0,))
  def _get_KKT(self, primals, duals, params):
    """Get KKT conditions and QP sub-problem params."""
    use_gn = self.opt.hess == "gn"

    # Get QP sub-problem params.
    dyn_params, cost_grads, constraint_vals, rollout_params = (
        self._get_step_params(primals, params)
    )

    # For Lagrangian stationarity, do adjoint recursion.
    grad_u, adjoints = self._compute_adjoint(
        primals, duals, dyn_params, cost_grads, constraint_vals
    )
    hess_params = self._get_QP_Hess(*primals, *duals, adjoints, use_gn, params)

    # Get objective
    obj = jnp.sum(
        self._costs(primals[1], optimizers.pad(primals[0]), *params[0])
    )

    Cu, _, Cx, _ = constraint_vals
    Yu, Yx = duals

    kkt_resid = {
        "obj": obj,
        "primal": (jnp.min(Cu), jnp.min(Cx)),
        "dual": (jnp.min(Yu), jnp.min(Yx)),
        "cslack": (jnp.max(jnp.abs(Cu * Yu)), jnp.max(jnp.abs(Cx * Yx))),
        "stat": jnp.max(jnp.abs(grad_u))
    }

    qp_params = (dyn_params, hess_params, cost_grads, constraint_vals,
                 rollout_params)
    return kkt_resid, qp_params

  @functools.partial(jax.jit, static_argnums=(0,))
  def _get_step_params(self, primals, params):
    """Get QP and Rollout params."""

    method = self.opt.method
    mu = self.opt.mu

    dyn_params, cost_grads, constraint_vals = self._get_QP_params(*primals,
                                                                  params)
    # Compute rollout gains
    rollout_params = ()
    if method in (SQP_METHOD.STABLE, SQP_METHOD.SENS, SQP_METHOD.APPROX_SENS):
      # Note: for SENS and APPROX_SENS, the LQR gains are used as backup only,
      # i.e., when the linesearch with the sensitivity-computed gains fails.
      K = self._get_LQR_control_gains(*primals, dyn_params, mu, params[0])
      rollout_params = (K,)

    return dyn_params, cost_grads, constraint_vals, rollout_params

  @functools.partial(jax.jit, static_argnums=(0,))
  def _check_opt(self, kkt, primals, duals):
    """Check satisfaction of optimality conditions."""

    primal_tol = self.opt.primal_tol
    dual_tol = self.opt.dual_tol

    U, _ = primals
    Yu, Yx = duals
    Y = jnp.hstack((optimizers.pad(Yu), Yx))

    tau_x = primal_tol * (1 + jnp.linalg.norm(U, "fro"))
    tau_d = dual_tol * (1 + jnp.linalg.norm(Y, "fro"))

    opt_feas = jnp.logical_and(
        jnp.minimum(kkt["primal"][0], kkt["primal"][1]) >= -tau_x,
        jnp.minimum(kkt["dual"][0], kkt["dual"][1]) >= -tau_d)
    opt_dual = jnp.logical_and(
        jnp.maximum(kkt["cslack"][0], kkt["cslack"][1]) <= tau_d,
        kkt["stat"] <= tau_d)

    return jnp.logical_and(opt_feas, opt_dual)

  def _do_step(self, primals, duals, Rho, qp_params, params, it):
    """Step function for SQP."""

    # Get QP params
    dyn_params, hess_params, cost_grads, constraint_vals, rollout_params = (
        qp_params
    )

    # Update and solve QP subproblem
    qp_solved, d_primals, duals_qp, self._qp_soln = self._QP.solve(
        self._qp_soln, *hess_params, *cost_grads, *dyn_params, *constraint_vals
    )

    assert qp_solved == 1, "QP not solved: {}".format(qp_solved)
    d_duals = tuple(Y_qp - Y for Y_qp, Y in zip(duals_qp, duals))

    # Compute slack and slack search direction
    slacks, d_slacks = self._compute_slacks(
        *constraint_vals, *d_primals, *duals, Rho
    )

    # Update penalty parameter
    search_dirs = (d_primals, d_duals, d_slacks)
    opt_vars = (primals, duals, slacks)

    # Using tree_util.Partial instead of functools.partial makes the
    # resulting function 'phi_and_grad' pytree compatible. This means, it can be
    # passed as an argument to a jittable function, i.e., self._linesearch.
    phi_and_grad = tree_util.Partial(self._ls_fnc, search_dirs, opt_vars,
                                     rollout_params)
    phi_0, phi_grad_0 = phi_and_grad(0., Rho, params)[:2]
    if self.opt.debug:
      print(f"phi(0):{phi_0}, phi_g(0):{phi_grad_0}")
    Rho = self._update_rho(phi_grad_0, Rho, cost_grads, hess_params, d_primals,
                           constraint_vals, slacks, duals, duals_qp)

    # Do DDP computations
    if self.opt.method in (SQP_METHOD.SENS, SQP_METHOD.APPROX_SENS):
      ddp_step = self._ddp_step(primals, qp_params, opt_vars, search_dirs,
                                Rho, params, it)
      if ddp_step[-3] > self.opt.ls_alpha_lb:
        # DDP step successful
        return ddp_step
      else:
        # DDP step failed, do stabilized step.
        ddp_errs = ddp_step[-1]
        if self.opt.verbose:
          print("Failed DDP step, executing Stabilized step.")
    else:
      ddp_errs = None

    # Do linesearch using stabilized method
    alpha_step, primals_up, duals_up = self._linesearch_jax(
        phi_and_grad, Rho, params)

    dU_step = primals_up[0] - primals[0]
    return primals_up, duals_up, Rho, alpha_step, dU_step, ddp_errs

  def _ddp_step(self, primals, qp_params, opt_vars, search_dirs, Rho, params,
                it):
    """Line-search using DDP method."""

    # Do the DDP computations
    d_primals = search_dirs[0]
    ddp_errs, rollout_params = self._DDP.backward_pass(it, qp_params, d_primals)
    if self.opt.verbose:
      print(f"|dU - dU_ddp|: {ddp_errs[0]}, |ddp_grad|: {ddp_errs[1]}")

    # Do line-search
    phi_and_grad = tree_util.Partial(self._ls_fnc, search_dirs, opt_vars,
                                     rollout_params)
    alpha_step, primals_up, duals_up = self._linesearch_jax(
        phi_and_grad, Rho, params)

    dU_step = primals_up[0] - primals[0]
    return primals_up, duals_up, Rho, alpha_step, dU_step, ddp_errs

  @functools.partial(jax.jit, static_argnums=(0,))
  def _compute_slacks(self, Cu, Ju, Cx, Jx, dU, dX, Yu, Yx, Rho):
    """Compute slack and slack search direction."""

    def get_s_ds(c, J, d, y, rho):
      s0 = jnp.maximum(jnp.zeros_like(c), c)
      s_ = jnp.maximum(jnp.zeros_like(c), c - (1.0/rho) * y)
      s = jnp.where(rho > 0., s_, s0)
      return s, c + J @ d - s
    vget_s_ds = jax.vmap(get_s_ds)

    Su, dS_u = vget_s_ds(Cu, Ju, dU, Yu, Rho[:-1])
    Sx, dS_x = vget_s_ds(Cx, Jx, dX, Yx, Rho)
    return (Su, Sx), (dS_u, dS_x)

  @functools.partial(jax.jit, static_argnums=(0,))
  def _ls_fnc(self, search_dirs, opt_vars, rollout_params, alpha, Rho, params):
    """Line-search function wrapper for merit and merit gradient."""
    phi_fnc = tree_util.Partial(self._phi, search_dirs, opt_vars,
                                rollout_params)

    (phi, phi_aux), phi_grad = jax.value_and_grad(
        phi_fnc, has_aux=True)(alpha, Rho, params)
    return phi, phi_grad, phi_aux[0], phi_aux[1]

  @functools.partial(jax.jit, static_argnums=(0,))
  def _phi(self, search_dirs, opt_vars, rollout_params, alpha, Rho, params):
    """Compute phi(alpha):= merit_fnc at updated (primal, dual, slacks)."""

    # Unpack
    d_primals, d_duals, d_slacks = search_dirs
    primals, duals, slacks = opt_vars

    # Get updated (primal, dual, slack) trajectories
    U_al, X_al = self._rollout(*primals, *d_primals, alpha, *rollout_params)
    Y_al = tuple(Y + alpha * dY for Y, dY in zip(duals, d_duals))
    S_al = tuple(S + alpha * dS for S, dS in zip(slacks, d_slacks))

    # Compute merit
    phi = self._merit_fn(U_al, X_al, *Y_al, *S_al, Rho, params)

    X_al = self._vec_wrap(X_al)
    return phi, ((U_al, X_al), Y_al)

  @functools.partial(jax.jit, static_argnums=(0,))
  def _update_rho(self, phi_grad_0, Rho, cost_grads, hess_params, d_primals,
                  constraint_vals, slacks, duals, duals_qp):
    """Update penalty parameter for merit function."""

    # Unpack
    q, r = cost_grads
    H, Q = hess_params
    dU, dX = d_primals
    Yu, Yx = duals
    Yu_qp, Yx_qp = duals_qp
    Cu, _, Cx, _ = constraint_vals
    Su, Sx = slacks

    # Compute optimal objective terms for QP solution
    def compute_quad(H_k, dx_k, du_k):
      dz_k = jnp.concatenate((dx_k, du_k))
      return dz_k @ (H_k @ dz_k)

    quad_dec = jnp.sum(jax.vmap(compute_quad)(H, dX[:-1], dU)) + dX[-1] @ (
        Q[-1] @ dX[-1])
    lin_dec = jnp.sum(q * dX) + jnp.sum(r * dU)

    Y = jnp.hstack((optimizers.pad(Yu), Yx))
    Y_qp = jnp.hstack((optimizers.pad(Yu_qp), Yx_qp))
    Eq = jnp.hstack((optimizers.pad(Cu - Su), Cx - Sx))
    norms = jnp.linalg.norm(Eq, 2, axis=1)
    active = jnp.greater(norms, 1e-6)
    nnz = jnp.sum(active)

    def update(_active, _Rho):
      Rho_hat = jnp.where(_active,
                          ((1. / nnz) * (lin_dec + 0.5 * quad_dec) + jnp.sum(
                              (2 * Y - Y_qp) * Eq, axis=1)) / jnp.square(norms),
                          jnp.zeros_like(_Rho))
      return jnp.where(_active, jnp.maximum(Rho_hat, 2 * _Rho), _Rho)

    return lax.cond(jnp.logical_or(phi_grad_0 <= -0.5 * quad_dec, nnz < 1),
                    lambda _active, _Rho: _Rho,
                    update, active, Rho)

  @functools.partial(jax.jit, static_argnums=(0,))
  def _linesearch_accept(self, phi_and_grad, Rho, params,
                         phi_0, phi_grad_0, alpha):
    """Accept conditions."""
    phi, phi_grad, primal, dual = phi_and_grad(alpha, Rho, params)
    cond_phi = phi <= (phi_0 + self.opt.ls_sigma * alpha * phi_grad_0)
    cond_grad = jnp.abs(phi_grad) <= -self.opt.ls_eta * phi_grad_0
    cond_grad_2 = phi_grad <= self.opt.ls_eta * phi_grad_0
    found = jnp.where(alpha < 0.99,
                      jnp.logical_and(cond_phi, cond_grad),
                      jnp.logical_and(cond_phi, jnp.logical_or(cond_grad,
                                                               cond_grad_2)))
    return found, primal, dual

  @functools.partial(jax.jit, static_argnums=(0,))
  def _linesearch_while_cond(self, inputs):
    """Continue predicate for search."""
    alpha, found, _, _ = inputs
    return jnp.logical_and(jnp.logical_not(found),
                           alpha >= self.opt.ls_alpha_lb)

  @functools.partial(jax.jit, static_argnums=(0,))
  def _linesearch_while_body(self, phi_and_grad, Rho, params,
                             phi_0, phi_grad_0, inputs):
    """Body function for search."""
    alpha, _, _, _ = inputs
    # Backtrack search interval: [beta_l, beta_u] * alpha.
    beta_u = self.opt.ls_beta
    beta_l = beta_u * beta_u
    alpha_u = beta_u * alpha
    alpha_l = beta_l * alpha
    # Use safe-guarded cubic interpolation.
    phi_u, phi_g_u, _, _ = phi_and_grad(alpha_u, Rho, params)
    phi_l, phi_g_l, _, _ = phi_and_grad(alpha_l, Rho, params)
    alpha_ = util.safe_cubic_opt(alpha_l, alpha_u,
                                 (phi_l, phi_g_l), (phi_u, phi_g_u))
    found, primal, dual = self._linesearch_accept(phi_and_grad, Rho, params,
                                                  phi_0, phi_grad_0, alpha_)
    return alpha_, found, primal, dual

  def _linesearch_python(self, phi_and_grad, Rho, params):
    return self._linesearch_template(util.while_loop, phi_and_grad, Rho, params)

  @functools.partial(jax.jit, static_argnums=(0,))
  def _linesearch_jax(self, phi_and_grad, Rho, params):
    return self._linesearch_template(lax.while_loop, phi_and_grad, Rho, params)

  def _linesearch_template(self, while_loop, phi_and_grad, Rho, params):
    """Perform backtracking linesearch with wrapped merit fnc (phi_and_grad)."""
    # Setup accept conditions
    phi_0, phi_grad_0, _, _ = phi_and_grad(0., Rho, params)
    # Run backtracking search
    alpha_0 = 1.
    found, primal, dual = self._linesearch_accept(
        phi_and_grad, Rho, params, phi_0, phi_grad_0, alpha_0)
    alpha_step, _, primal_up, dual_up = while_loop(
        self._linesearch_while_cond,
        functools.partial(self._linesearch_while_body,
                          phi_and_grad, Rho, params, phi_0, phi_grad_0),
        (alpha_0, found, primal, dual))
    return alpha_step, primal_up, dual_up

  def _record_summary(self, history, alpha, kkt_resid, ddp_errs):
    """Record iteration summary."""
    # Record summary.
    history["steplength"].append(alpha)
    history["obj"].append(kkt_resid["obj"])
    history["min_viol"].append(kkt_resid["primal"][1])
    if ddp_errs:
      history["ddp_err"].append(ddp_errs[0])
      history["ddp_err_grad"].append(ddp_errs[1])
