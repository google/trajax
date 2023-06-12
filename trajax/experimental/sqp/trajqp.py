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
"""QP sub-problems for Shooting SQP."""

import abc
import functools

import cvxpy as cp
import jax
from jax import device_put
import jax.numpy as jnp
import numpy as np
from trajax import optimizers


cvxpy_solvers = {
    "ecos": cp.ECOS,
    "cvxopt": cp.CVXOPT
}


class TrajQP(abc.ABC):
  """Class for SQP QP sub-problem.

  Creates and solves the following parametric QP problem:
    min sum_k   .5* dz_k'Z_k dz_k + q_k dx_k + r_k du_k +
               +.5*dx_T Z_T dx_T + q_T dx_T
    s.t. dx_{k+1} = A_k dx_k + B_k du_k k=0,...,T-1
         cu_k + Ju_k du_k >= 0  k=0,...,T-1
         cx_k + Jx_k dx_k >= 0  k=1,...,T
  where:
    dz_k = (dx_k, du_k)
    Z_k = [Z_xx, Z_xu;
           Z_ux, Z_uu]_k
    Z_T = Z_xx_T
  see https://arxiv.org/pdf/2109.07081.pdf for definitions.
  """

  def __init__(self, n: int, m: int, T: int, n_cu: int, n_cx: int,
               **qp_options):
    """Initialize the problem.

    Args:
      n: state dimension (int, >=1)
      m: control dimension (int, >=1)
      T: horizon (int, >= 2)
      n_cu: per timestep # of control constraints (int)
      n_cx: per timestep # of state constraints (int)
      **qp_options: options for qp solver.
    """
    self._n = n
    self._m = m
    self._T = T
    self._n_cu = n_cu
    self._n_cx = n_cx
    self._options = qp_options

    self.soln = jnp.zeros((T, m))

  def reset_soln(self):
    """Resets default QP solution."""
    self.soln = jax.tree_map(jnp.zeros_like, self.soln)
    return self.soln

  @abc.abstractmethod
  def _update_params(self, *qp_params):
    """Updates QP Parameters."""

  @abc.abstractmethod
  def solve(self, prev_soln, *updated_params):
    """Solves QP with updated params.

    Args:
      prev_soln: previous solution (for warm-starting).
      *updated_params: updated parameters for QP:
        - Z: [T, n+m, n+m] ndarray of the Hessian matrices {Z_k}, k: [0,T-1].
        - Q: [T+1, n, n] ndarray of the 'xx' portion of the Hessian matrices.
        - q: [T+1, n] ndarray of cost state gradients, k: [0,T].
        - r: [T, m] ndarray of cost control gradients, k: [0,T-1].
        - A, B: [T, n, n], [T, n, m] ndarrays of dynamics Jacobians, k: [0,T-1].
        - Cu: [T, n_cu] ndarray of control constraints, k: [0,T-1].
        - Ju: [T, n_cu, m] ndarray of control constraint Jacobians, k: [0,T-1].
        - Cx: [T+1, n_cx] ndarray of state constraints, k: [0,T-1].
        - Jx: [T+1, n_cx, n] ndarray of state constraint Jacobians, k: [0,T-1].
        (Note: Cx[0] and Jx[0] not used.)
    Returns:
      status: int, 1="solved"
      d_primals: primal soln; tuple of (dU: [T, m], dX: [T+1, n])-ndarrays.
      duals_qp: dual soln; tuple of (Yu: [T, n_cu], Yx: [T+1, n_cx])-ndarrays.
      soln: raw solution object - used to warm-start next iteration.
    """


class TrajQPcvx(TrajQP):
  """Cvxpy version of TrajQP."""

  def __init__(self, n: int, m: int, T: int, n_cu: int, n_cx: int, solver: str,
               **qp_options):

    """Initialize the problem."""
    assert solver in cvxpy_solvers.keys()

    super().__init__(n, m, T, n_cu, n_cx, **qp_options)
    self._solver = cvxpy_solvers[solver]

    # Initialize parameters
    self._init_params(n, m, n_cu, n_cx, T)
    # Setup parametrized cvx problem
    self._create_prob()

  def _init_params(self, n, m, n_cu, n_cx, T):
    """Initialize parameters for the QP Problem."""

    self._q = cp.Parameter((T+1, n))
    self._r = cp.Parameter((T, m))
    self._Cu = cp.Parameter((T, n_cu))
    self._Cx = cp.Parameter((T+1, n_cx))
    self._ZT = cp.Parameter((n, n), PSD=True)
    Z, A, B, Ju, Jx = {}, {}, {}, {}, {}
    for j in range(T):
      Z[j] = cp.Parameter((n+m, n+m), PSD=True)
      A[j] = cp.Parameter((n, n))
      B[j] = cp.Parameter((n, m))
      Ju[j] = cp.Parameter((n_cu, m))
      Jx[j] = cp.Parameter((n_cx, n))
    Jx[T] = cp.Parameter((n_cx, n))
    self._Z = Z
    self._A = A
    self._B = B
    self._Ju = Ju
    self._Jx = Jx

  def _create_prob(self):
    """Create the QP Problem."""

    T, n, m = self._T, self._n, self._m

    # Variables
    self._dX = cp.Variable((T, n))  # k=1,...,T
    self._dU = cp.Variable((T, m))  # k=0,...,T-1

    # Cost expression
    cost = 0.

    # Constraints
    self._constr_dyn = []
    self._constr_ineq_u = []
    self._constr_ineq_x = []

    for j in range(T):
      if j == 0:
        dx = np.zeros((n,))
      else:
        dx = self._dX[j-1]
      dz = cp.hstack([dx, self._dU[j]])
      cost += (1 / 2) * cp.quad_form(
          dz, self._Z[j]) + self._q[j] @ dx + self._r[j] @ self._dU[j]

      self._constr_ineq_u += [self._Cu[j] + self._Ju[j] @ self._dU[j] >= 0.]
      if j > 0:
        self._constr_ineq_x += [self._Cx[j] + self._Jx[j] @ dx >= 0.]
      self._constr_dyn += [
          self._dX[j] == self._A[j] @ dx + self._B[j] @ self._dU[j]
      ]

    cost += (1 / 2) * cp.quad_form(self._dX[-1],
                                   self._ZT) + self._q[-1] @ self._dX[-1]
    self._constr_ineq_x += [self._Cx[-1] + self._Jx[T] @ self._dX[-1] >= 0.]

    constr = self._constr_dyn + self._constr_ineq_u + self._constr_ineq_x

    self._prob = cp.Problem(cp.Minimize(cost), constr)

  def _update_params(self, *qp_params):
    """Update QP Parameters."""

    # Convert to regular arrays
    Z, Q, q, r, A, B, Cu, Ju, Cx, Jx = (np.array(arr) for arr in qp_params)

    # Update
    self._q.value = q
    self._r.value = r
    self._Cu.value = Cu
    self._Cx.value = Cx
    self._ZT.value = Q[-1]

    for j in range(self._T):
      self._Z[j].value = Z[j]
      self._A[j].value = A[j]
      self._B[j].value = B[j]
      self._Ju[j].value = Ju[j]
      self._Jx[j].value = Jx[j]
    self._Jx[self._T].value = Jx[-1]

  def solve(self, prev_soln, *updated_params):
    """Solve QP with updated parameters."""

    del prev_soln  # cvx doesn't use warm-starting
    self._update_params(*updated_params)
    self._prob.solve(self._solver, **self._options)

    if self._prob.status not in ("infeasible", "unbounded"):
      dU_d = device_put(self._dU.value)
      dX_d = jnp.vstack((jnp.zeros((self._n,)),
                         device_put(self._dX.value)))
      Yu = jnp.vstack([
          device_put(constr_u.dual_value) for constr_u in self._constr_ineq_u])
      Yx = jnp.vstack([
          device_put(constr_x.dual_value) for constr_x in self._constr_ineq_x])
      Yx = jnp.vstack((jnp.zeros_like(Yx[0]), Yx))  # assumes Cx[0] > 0.
      return 1, (dU_d, dX_d), (Yu, Yx), dU_d
    else:
      return self._prob.status, None, None, None


class TrajQPalqr(TrajQP):
  """Use augmented iLQR to solve QP problem."""

  def __init__(self, n: int, m: int, T: int, n_cu: int, n_cx: int,
               **qp_options):
    default_options = {
        "constraints_threshold": 1e-3,
        "penalty_update_rate": 5,
        "maxiter_al": 10
    }
    default_options.update(qp_options)
    super().__init__(n, m, T, n_cu, n_cx, **default_options)

  def _qp_cost(self, dx, du, k, Z, ZT, q, r):
    """Defines QP cost function."""
    dz = jnp.concatenate((dx, du))
    stage_cost = 0.5 * jnp.vdot(dz, Z[k] @ dz) + jnp.vdot(
        q[k], dx) + jnp.vdot(r[k], du)
    term_cost = 0.5 * jnp.vdot(dx, ZT @ dx) + jnp.vdot(q[-1], dx)
    return jnp.where(k == self._T, term_cost, stage_cost)

  def _qp_cons(self, dx, du, k, Cu, Ju, Cx, Jx):
    """Defines QP stage-wise constraint."""
    state_cons = jnp.where(k == 0, -jnp.ones(self._n_cx),
                           -(Cx[k] + Jx[k] @ dx))
    control_cons = jnp.where(k == self._T, -jnp.ones(self._n_cu),
                             -(Cu[k] + Ju[k] @ du))
    return jnp.concatenate((state_cons, control_cons))

  def _lin_dyn(self, dx, du, k, A, B):
    """Defines QP sub-problem linear dynamics."""
    return A[k] @ dx + B[k] @ du

  def _update_params(self, *qp_params):
    """Updates QP params."""
    Z, Q, q, r, A, B, Cu, Ju, Cx, Jx = qp_params

    cost_params = {
        "Z": optimizers.pad(Z),
        "ZT": Q[-1],
        "q": q,
        "r": optimizers.pad(r),
    }
    cons_params = {
        "Cu": optimizers.pad(Cu),
        "Ju": optimizers.pad(Ju),
        "Cx": Cx,
        "Jx": Jx
    }
    dyn_params = {
        "A": A,
        "B": B
    }
    return cost_params, cons_params, dyn_params

  @functools.partial(jax.jit, static_argnums=(0,))
  def solve(self, dU_0, *updated_params):
    """Solves QP sub-problem using AL-iLQR."""
    # First, update cost and constraint w/ new params.
    cost_params, cons_params, dyn_params = self._update_params(*updated_params)
    dyn = functools.partial(self._lin_dyn, **dyn_params)
    cost = functools.partial(self._qp_cost, **cost_params)
    ineq_cons = functools.partial(self._qp_cons, **cons_params)

    dx_0 = jnp.zeros(self._n)

    results = optimizers.constrained_ilqr(cost, dyn, dx_0, dU_0,
                                          inequality_constraint=ineq_cons,
                                          **self._options)
    return self._parse_results(results)

  def _parse_results(self, results):
    """Parses results from AL-iLQR."""

    dX, dU, _, Y_qp, *_ = results
    Yx, Yu = jnp.split(Y_qp, [self._n_cx], axis=1)
    Yu = Yu[:-1]

    return 1, (dU, dX), (Yu, Yx), dU
