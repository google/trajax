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
"""Header class for Trajectory Optimization solvers."""

import functools

from typing import Optional, Tuple

import jax
from jax import lax
from jax import tree_util
import jax.numpy as jnp
from trajax import optimizers
from trajax import tvlqr

from trajax.experimental.sqp import util


STATE = jnp.ndarray
CONTROL = jnp.ndarray


class TrajectoryOptimizationSolver(object):
  """Header class for constrained trajectory optimization solvers.

  NOTE: This class packages together useful sub-routines for various constrained
  trajectory optimization problem solvers, e.g., SQP. As a result, all methods
  and members are protected, with the expectation they can only be accessed
  by a specific inherited solver.
  """

  def __init__(
      self, n: int, m: int, T: int,
      dynamics, cost, control_bounds: Tuple[CONTROL, CONTROL],
      state_constraint, s1_ind: Optional[Tuple[int, ...]] = None):
    """Initialize problem structure.

    Args:
      n: state dimension (int, >=1)
      m: control dimension (int, >=1)
      T: horizon (int, >= 2)
      dynamics: Callable, (x:(n,) array, u:(m,) array, t:int) --> (n,) array.
      cost: Callable, (x:(n,) array, u:(m,) array, t:int, *cost_args) --> float.
      control_bounds: tuple of (u_lower: (m,) ndarray, u_upper: (m,) ndarray),
        defining bounds on control; use jnp.inf for unconstrained components.
      state_constraint: Callable,
        (x:(n,) array, t:int, *cons_args) --> (n_cx,) array; want >= 0.
      s1_ind: Tuple of indices indicating components of state lying on S1.
    """
    # Store dimensions.
    self._n = n
    self._m = m
    self._T = T
    self._timesteps = jnp.arange(T + 1)

    self._state_wrap = util.get_s1_wrapper(s1_ind)
    self._vec_wrap = jax.vmap(self._state_wrap)

    control_constraint = self._get_control_constraint(control_bounds)
    c_u = control_constraint(jnp.zeros((m,)), 0)
    self._n_cu = c_u.shape[0]
    c_x = state_constraint(jnp.zeros((n,)), 0)
    self._n_cx = c_x.shape[0]

    # Dynamics.
    self._dynamics = dynamics

    ## Define some useful vectorized functions.
    @jax.jit
    def lagrangian(x, u, t, y_u, y_x, params):
      """Stage-wise Lagrangian."""
      cost_p, cons_p = params
      C = cost(x, u, t, *cost_p)
      c_u = control_constraint(u, t)
      c_u = jnp.where(t == T, jnp.zeros_like(c_u), c_u)
      c_x = state_constraint(x, t, *cons_p)
      L = C - jnp.vdot(c_u, y_u) - jnp.vdot(c_x, y_x)
      return L

    @jax.jit
    def hamiltonian(x, u, t, y_u, y_x, nu, params):
      """Stage-wise Hamiltonian."""
      L = lagrangian(x, u, t, y_u, y_x, params)
      H = L + jnp.vdot(dynamics(x, u, t), nu)
      return jnp.where(t == T, L, H)

    # Quadratize Hamiltonian, Lagrangian, and Cost.
    self._quadratizer_H = optimizers.quadratize(hamiltonian, argnums=6)
    self._quadratizer_L = optimizers.quadratize(lagrangian, argnums=5)
    self._quadratizer_C = optimizers.quadratize(cost, argnums=3)

    # Hessian function
    def create_Hessian(Q_k, R_k, M_k):
      return jnp.block([[Q_k, M_k], [M_k.T, R_k]])
    self._vec_Hessian = jax.vmap(create_Hessian)
    self._psd = jax.vmap(optimizers.project_psd_cone, in_axes=(0, None))

    # Vectorized costs and Gradients
    self._costs = functools.partial(optimizers.evaluate, cost)
    self._cost_gradients = optimizers.linearize(cost)

    # Dynamics linearization
    self._dynamics_jacobians = optimizers.linearize(dynamics)

    # Constraints
    self._control_bounds = control_bounds
    self._u_con = optimizers.vectorize(control_constraint, argnums=2)
    self._Ju_con = optimizers.vectorize(jax.jacobian(control_constraint),
                                        argnums=2)
    self._x_con = optimizers.vectorize(state_constraint, argnums=2)
    self._Jx_con = optimizers.vectorize(jax.jacobian(state_constraint),
                                        argnums=2)

  def _check_primals(self, U, X):
    """Check shape of primals."""
    assert U.shape == (self._T, self._m)
    assert X.shape == (self._T + 1, self._n)

  def _check_duals(self, Yu, Yx):
    """Check shape of primals."""
    assert Yu.shape == (self._T, self._n_cu)
    assert Yx.shape == (self._T + 1, self._n_cx)

  def _get_control_constraint(self, control_bounds):
    """Create control constraint function, c_u(u_t, t) >= 0, from bounds."""
    u_lower, u_upper = control_bounds
    lower_ind = jnp.logical_not(jnp.isinf(u_lower))
    upper_ind = jnp.logical_not(jnp.isinf(u_upper))

    def control_constraint(u, t):
      """c(u, t) >= 0 representation."""
      del t
      # Lower bound: u(j) >= u_lower(j)
      c_lower = u[lower_ind] - u_lower[lower_ind]
      # Upper bound: u_upper(j) >= u(j)
      c_upper = u_upper[upper_ind] - u[upper_ind]
      return jnp.concatenate((c_lower, c_upper))

    return jax.jit(control_constraint)

  @functools.partial(jax.jit, static_argnums=(0, 6))
  def _get_QP_Hess(self, U, X, Yu, Yx, Nu, use_gauss_newton, params):
    """Get stage-wise Hessians of Hamiltonian (or Lagrangian).

    Args:
      U: control trajectory, ndarray: [T, m]
      X: state trajectory, ndarray: [T+1, n]
      Yu: control constraint dual variable trajectory, ndarray: [T, n_cu]
      Yx: state constraint dual variable trajectory, ndarray: [T+1, n_cx]
      Nu: Hessian adjoint trajectory, ndarray:[T, n]
      use_gauss_newton: Boolean indicating use of Gauss-Newton Hessian.
      params: tuple of cost & state constraint parameter tuples, where first
        element of constraint parameter tuple is the current state trajectory.

    Returns:
      tuple of (H, Q), where H: [T, n+m, n+m]-ndarray is the Hessian trajectory,
      and Q: [T+1, n, n]-ndarray is the state block of the Hessian trajectory.
    """
    self._check_primals(U, X)
    self._check_duals(Yu, Yx)
    assert Nu.shape == (self._T, self._n)

    if use_gauss_newton:
      Q, R, M = self._quadratizer_L(X, optimizers.pad(U), self._timesteps,
                                    optimizers.pad(Yu), Yx, params)
    else:
      Q, R, M = self._quadratizer_H(X, optimizers.pad(U), self._timesteps,
                                    optimizers.pad(Yu), Yx, optimizers.pad(Nu),
                                    params)

    Q = self._psd(Q, 0.)
    R = self._psd(R, 1.e-3)

    H = self._vec_Hessian(Q[:-1], R[:-1], M[:-1])  # [T, n+m, n+m]
    H = self._psd(H, 1.e-3)

    return (H, Q)

  @functools.partial(jax.jit, static_argnums=(0,))
  def _get_QP_params(self, U, X, params):
    """Get params for QP sub-problem.

    Args:
      U: control trajectory, ndarray: [T, m]
      X: state trajectory, ndarray: [T+1, n]
      params: tuple of cost & state constraint parameter tuples, where first
        element of constraint parameter tuple is the current state trajectory.

    Returns:
      dyn_params: tuple of dynamics Jacobians,
          (A: [T, n, n]-ndarray, B: [T, n, m]-ndarray)
      cost_grads: tuple of cost gradients,
          (q: [T+1, n]-ndarray, r: [T, m]-ndarray)
      constraint_vals: tuple of constraint & gradient trajectories,
        (Cu: [T, n_cu]-ndarray, Ju: [T, n_cu, m]-ndarray,
         Cx: [T+1, n_cx]-ndarray, Jx: [T+1, n_cx, n]-ndarray)
    """
    self._check_primals(U, X)

    cost_p, cons_p = params
    # Gradients
    q, r = self._cost_gradients(X, optimizers.pad(U), self._timesteps, *cost_p)

    # Dynamics
    A, B = self._dynamics_jacobians(X[:-1], U, self._timesteps[:-1])

    # Constraints
    Cu = self._u_con(U, self._timesteps[:-1])
    Ju = self._Ju_con(U, self._timesteps[:-1])
    Cx = self._x_con(X, self._timesteps, *cons_p)
    Jx = self._Jx_con(X, self._timesteps, *cons_p)

    return (A, B), (q, r[:-1]), (Cu, Ju, Cx, Jx)

  @functools.partial(jax.jit, static_argnums=(0,))
  def _merit_fn(self, U, X, Yu, Yx, Su, Sx, Rho, params):
    """Compute merit function.

    Args:
      U: control trajectory, ndarray: [T, m]
      X: state trajectory, ndarray: [T+1, n]
      Yu: control constraint dual variable trajectory, ndarray: [T, n_cu]
      Yx: state constraint dual variable trajectory, ndarray: [T+1, n_cx]
      Su: control constraint slack variable trajectory, ndarray: [T, n_cu]
      Sx: state constraint slack variable trajectory, ndarray: [T+1, n_cx]
      Rho: vector of penalty parameters, ndarray: [T+1,]
      params: tuple of cost & state constraint parameter tuples, where first
        element of constraint parameter tuple is the current state trajectory.

    Returns:
      scalar merit function value.
    """
    self._check_primals(U, X)
    self._check_duals(Yu, Yx)
    self._check_duals(Su, Sx)

    # Get terms
    cost_p, cons_p = params
    costs = self._costs(X, optimizers.pad(U), *cost_p)
    Cu = self._u_con(U, self._timesteps[:-1])
    Cx = self._x_con(X, self._timesteps, *cons_p)
    eq_resid = (optimizers.pad(Cu - Su), Cx - Sx)

    # Compute merit
    duals = (optimizers.pad(Yu), Yx)
    resid_merit = tree_util.tree_reduce(
        jnp.add,
        tree_util.tree_map(
            lambda resid, Y: jnp.sum((jnp.diag(Rho / 2.0) @ resid - Y) * resid),
            eq_resid, duals))
    return jnp.sum(costs) + resid_merit

  @functools.partial(jax.jit, static_argnums=(0, 6))
  def _compute_adjoint(self, primals, duals, dynamics_params, cost_grads,
                       constraint_vals):
    """Compute grad_u(Lag) and adjoint of augmented Lagrangian.

    Args:
      primals: tuple of (U, X) trajectories; see above.
      duals: tuple of (Yu, Yx) trajectories; see above.
      dynamics_params: tuple of dynamics Jacobians, see above.
      cost_grads: tuple of cost gradients, see above.
      constraint_vals: tuple of constraint & gradients, see above.

    Returns:
      grad_u: gradient(Lagrangian), ndarray: [T, m].
      hess_params: tuple output from _get_QP_Hess.
    """
    self._check_primals(*primals)
    self._check_duals(*duals)

    A, B = dynamics_params
    q, r = cost_grads
    _, Ju, _, Jx = constraint_vals
    Yu, Yx = duals

    nu_N = q[-1] - Jx[-1].T @ Yx[-1]
    def dyn_costate(nu_kp1, k):
      grad_u_lhat = r[k] - Ju[k].T @ Yu[k]
      grad_x_lhat = q[k] - Jx[k].T @ Yx[k]
      grad_u_k = grad_u_lhat + B[k].T @ nu_kp1
      nu_k = grad_x_lhat + A[k].T @ nu_kp1

      return nu_k, (grad_u_k, nu_kp1)

    grad_u, Nu_hess = lax.scan(dyn_costate, nu_N,
                               jnp.arange(self._T - 1, -1, -1))[1]
    grad_u = jnp.flipud(grad_u)
    Nu_hess = jnp.flipud(Nu_hess)

    return grad_u, Nu_hess

  @functools.partial(jax.jit, static_argnums=(0,))
  def _get_LQR_control_gains(self, U, X, dynamics_params, mu, cost_params):
    """Get control gains for stabilized rollout.

    Args:
      U: control trajectory, see above.
      X: state trajectory, see above.
      dynamics_params: tuple of dynamics Jacobians, see above.
      mu: float tuning parameter for control cost penalty, in range (0, 1].
      cost_params: tuple of cost function parameters.

    Returns:
      K: sequence of control gains, ndarray: [T, m, n].
    """
    self._check_primals(U, X)

    Q, R, M = self._quadratizer_C(X, optimizers.pad(U), self._timesteps,
                                  *cost_params)

    Q = self._psd(Q, 1.0)
    R = mu * self._psd(R[:-1], 1.e-3)

    K, _, _, _ = tvlqr.tvlqr(Q, jnp.zeros((self._T+1, self._n)),
                             R, jnp.zeros((self._T, self._m)),
                             M, *dynamics_params, jnp.zeros((self._T, self._n)))
    return K
