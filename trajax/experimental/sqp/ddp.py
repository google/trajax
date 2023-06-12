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
# Lint as: python3
# pylint: disable=invalid-name
"""Sensitivity-based DDP gain computation for SQP."""

import functools

import jax
import jax.numpy as jnp
from trajax import optimizers
from trajax import tvlqr

from trajax.experimental.sqp import trajqp as qp


class DDP_QP:
  """Get Sensitivity gains using parallelized iLQR.

  Differentiate unconstrained version of QP sub-problem (from trajqp.TrajQP):
    min sum_k   .5* dz_k'Z_k dz_k + q_k dx_k + r_k du_k +
                 +.5*dx_T Z_T dx_T + q_T dx_T
      s.t. dx_{k+1} = A_k dx_k + B_k du_k k=0,...,T-1
           cu_k + Ju_k du_k >= 0  k=0,...,T-1
           cx_k + Jx_k dx_k >= 0  k=1,...,T
      Add: dx_k = dx_k_qp  k=0,...,T-1
  """

  def __init__(self, n: int, m: int, T: int, n_cu: int, n_cx: int, **options):
    """Initialize the problem.

    Args:
      n: state dimension (int, >=1)
      m: control dimension (int, >=1)
      T: horizon (int, >= 2)
      n_cu: per timestep # of control constraints (int)
      n_cx: per timestep # of state constraints (int)
      **options: configurable parameters.
    """

    self._n = n
    self._dx_0 = jnp.zeros((n,))
    self._T = T
    self._n_cx = n_cx

    default_options = {
        "ddp_gamma": 1e-3,  # initial DDP gamma parameter; range: (1e-5, 1e-2).
        "ddp_gamma_ratio": 1.0,  # update ratio for gamma; range: (1e-1, 1].
        "ddp_maxiter": 100,  # maxiter for iLQR feedback gain solves.
        "ddp_eps": 1e-6,  # log-barrier threshold for iLQR constraints.
    }
    default_options.update(options)
    self._options = default_options

    qp_prob = qp.TrajQPalqr(n, m, T, n_cu, n_cx)
    self._qp_cost = qp_prob._qp_cost
    self._qp_cons = qp_prob._qp_cons
    self._lin_dyn = qp_prob._lin_dyn

  @functools.partial(jax.jit, static_argnums=(0,))
  def _cost_aug(self, dx, du, t,
                Z, ZT, q, r, Cu, Ju, Cx, Jx, gamma):
    """Define inequality-constraint-augmented QP cost."""

    reg_cost = self._qp_cost(dx, du, t, Z, ZT, q, r)
    ineq_cons = -self._qp_cons(dx, du, t, Cu, Ju, Cx, Jx)
    cost_ineq = -gamma * jnp.sum(jnp.log(ineq_cons))
    return reg_cost + cost_ineq

  @functools.partial(jax.jit, static_argnums=(0,))
  def _eq_aug_cost(self, dx_qp_k, k, dx, du, t,
                   Z, ZT, q, r, Cu, Ju, Cx, Jx, gamma):
    """Define equality-constraint-augmented QP cost."""
    aug_cost = self._cost_aug(dx, du, t, Z, ZT, q, r, Cu, Ju, Cx, Jx, gamma)
    cost_eq = 0.5 * (1/gamma) * jnp.vdot(dx - dx_qp_k, dx - dx_qp_k)
    return jnp.where(t == k, aug_cost + cost_eq, aug_cost)

  @functools.partial(jax.jit, static_argnums=(0,))
  def _adjust_cons(self, cu, Ju, cx, Jx, du, dx):
    """Adjusts constraints to ensure well-posedness of log-barrier."""

    resid_u = cu + Ju @ du
    resid_x = cx + Jx @ dx

    eps = self._options["ddp_eps"]

    cu_ = jnp.where(jnp.greater_equal(resid_u, eps), cu,
                    cu - resid_u + eps)
    cx_ = jnp.where(jnp.greater_equal(resid_x, eps), cx,
                    cx - resid_x + eps)
    return cu_, cx_

  @functools.partial(jax.jit, static_argnums=(0,))
  def backward_pass(self, itr, qp_params, d_primals):
    """Get sensitivity gains.

    Args:
      itr: SQP iteration.
      qp_params: QP sub-problem params, see shootsqp.ShootSQP._do_step().
      d_primals: tuple of (dU, dX) solution from QP solve.

    Returns:
      ddp_err: Tuple of:
               - dU_err: max 2-norm error between QP and iLQR solutions.
               - grad_lqr: maximum iLQR gradient inf norm, ideally 0.
      gains: ([T, m, n]- ndarray, ) corresponding to rollout gains.
    """

    # Update barrier parameter.
    gamma_0 = self._options["ddp_gamma"]
    g_ratio = self._options["ddp_gamma_ratio"]
    gamma = jnp.maximum(gamma_0 * (g_ratio ** itr), 1e-5)

    # Unpack
    dyn_params, hess_params, cost_grads, constraint_vals, _ = qp_params

    # Define Linear dynamics.
    A, B = dyn_params
    lin_dyn = functools.partial(self._lin_dyn, A=A, B=B)
    dX_qp = optimizers.rollout(lin_dyn, d_primals[0], self._dx_0)

    # Adjust constraint params to ensure log-barrier well-posed.
    Cu, Ju, Cx, Jx = constraint_vals
    Cu_, Cx_ = jax.vmap(self._adjust_cons)(
        optimizers.pad(Cu), optimizers.pad(Ju), Cx, Jx,
        optimizers.pad(d_primals[0]), dX_qp)

    qp_args = {
        "Z": optimizers.pad(hess_params[0]),
        "ZT": hess_params[1][-1],
        "q": cost_grads[0],
        "r": optimizers.pad(cost_grads[1]),
        "Cu": Cu_,
        "Ju": optimizers.pad(Ju),
        "Cx": Cx_,
        "Jx": Jx,
        "gamma": gamma
    }

    # Define QP augmented cost.
    cost_uncon = functools.partial(self._eq_aug_cost, **qp_args)

    # Define step-k unconstrained LQR problem.
    def calibrate_ilqr(dx_qp_k, k):
      def get_du_k(dx_opt):
        _, dU, _, ilqr_grad, *_ = optimizers.ilqr(
            functools.partial(cost_uncon, dx_opt, k),  # fnc only of (dx, du, t)
            lin_dyn,
            self._dx_0, d_primals[0],
            maxiter=self._options["ddp_maxiter"],
        )
        return dU[k], (dU[k], jnp.max(jnp.abs(ilqr_grad)))

      Ku_k, aux_k = jax.jacobian(get_du_k, has_aux=True)(dx_qp_k)
      du_k, lqr_k_grad = aux_k
      return du_k, lqr_k_grad, Ku_k

    # Solve in parallel across all timesteps.
    dU_lqr, Grad_lqr, K_ddp = jax.vmap(calibrate_ilqr)(
        dX_qp[:-1], jnp.arange(self._T))

    # Compute error metric
    dU_err = jnp.max(jnp.linalg.norm(dU_lqr - d_primals[0], 2, axis=1))
    return (dU_err, jnp.max(Grad_lqr)), (K_ddp,)


class APPROX_DDP_QP(DDP_QP):
  """Approximation of DDP_QP using one iLQR+TV-LQR solve."""

  @functools.partial(jax.jit, static_argnums=(0,))
  def backward_pass(self, itr, qp_params, d_primals):
    """Simplified backward pass computation."""
    del itr

    # Unpack
    dyn_params, hess_params, cost_grads, constraint_vals, _ = qp_params

    # Define Linear dynamics.
    A, B = dyn_params
    lin_dyn = functools.partial(self._lin_dyn, A=A, B=B)
    dX_qp = optimizers.rollout(lin_dyn, d_primals[0], self._dx_0)

    # Adjust constraint params to ensure log-barrier well-posed.
    Cu, Ju, Cx, Jx = constraint_vals
    Cu_, Cx_ = jax.vmap(self._adjust_cons)(
        optimizers.pad(Cu), optimizers.pad(Ju), Cx, Jx,
        optimizers.pad(d_primals[0]), dX_qp)

    qp_args = {
        "Z": optimizers.pad(hess_params[0]),
        "ZT": hess_params[1][-1],
        "q": cost_grads[0],
        "r": optimizers.pad(cost_grads[1]),
        "Cu": Cu_,
        "Ju": optimizers.pad(Ju),
        "Cx": Cx_,
        "Jx": Jx,
        "gamma": self._options["ddp_gamma"]
    }

    # Define QP augmented cost.
    cost_uncon = functools.partial(self._cost_aug, **qp_args)

    # Solve iLQR to get the last iteration's tv-lqr params
    results = optimizers.ilqr(
        cost_uncon, lin_dyn,
        self._dx_0, d_primals[0],
        maxiter=self._options["ddp_maxiter"])

    dU, ilqr_grad, lqr = results[1], results[3], results[-2]

    # Get the Taylor expansions of the iLQR cost
    Q, q, R, r, M, _, _ = lqr
    # Compute TV-LQR gains
    K_ddp, _, _, _ = tvlqr.tvlqr(Q, jnp.zeros_like(q),
                                 R, jnp.zeros_like(r),
                                 M, A, B, jnp.zeros((self._T, self._n)))
    dU_err = jnp.max(jnp.linalg.norm(dU - d_primals[0], 2, axis=1))
    return (dU_err, jnp.max(jnp.abs(ilqr_grad))), (K_ddp,)
