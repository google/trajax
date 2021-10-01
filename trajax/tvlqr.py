# Copyright 2021 Google LLC
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
"""JAX Solver for Discrete-time Finite Horizon Time-varying LQR.

Solve:

 min_{x0, x1, ...xT, u0, u1...u_{T-1}}

      sum_{t=0}^{T-1} [0.5* x(t)^T Q(t) x(t) + q(t)^T x(t) +
                      0.5* u(t)^T R(t) u(t) + r(t)^T u(t) +
                            +
                      x(t)^T M(t) u(t)]
                            +
                      0.5*x(T)^T Q(T) x(t) + q(T)^T x(T)

      subject to Linear Dynamics:
                      x(t+1) = A(t) x(t) + B(t) u(t) + c(t)

"""
from functools import partial  # pylint: disable=g-importing-member

from jax import jit
from jax import lax
from jax import ops
import jax.numpy as np
import jax.scipy as sp


@jit
def rollout(K, k, x0, A, B, c):
  """Rolls-out time-varying linear policy u[t] = K[t] x[t] + k[t]."""

  T, m, n = K.shape
  X = np.zeros((T + 1, n))
  U = np.zeros((T, m))
  X = X.at[0].set(x0)

  def body(t, inputs):
    X, U = inputs
    u = np.matmul(K[t], X[t]) + k[t]
    x = np.matmul(A[t], X[t]) + np.matmul(B[t], u) + c[t]
    X = X.at[t + 1].set(x)
    U = U.at[t].set(u)
    return X, U

  return lax.fori_loop(0, T, body, (X, U))


@jit
def lqr_step(P, p, Q, q, R, r, M, A, B, c, delta=1e-8):
  """Single LQR Step.

  Args:
    P: [n, n] numpy array.
    p: [n] numpy array.
    Q: [n, n] numpy array.
    q: [n] numpy array.
    R: [m, m] numpy array.
    r: [m] numpy array.
    M: [n, m] numpy array.
    A: [n, n] numpy array.
    B: [n, m] numpy array.
    c: [n] numpy array.
    delta: Enforces positive definiteness by ensuring smallest eigenval > delta.

  Returns:
    P, p: updated matrices encoding quadratic value function.
    K, k: state feedback gain and affine term.
  """
  symmetrize = lambda x: (x + x.T) / 2

  AtP = np.matmul(A.T, P)
  AtPA = symmetrize(np.matmul(AtP, A))
  BtP = np.matmul(B.T, P)
  BtPA = np.matmul(BtP, A)

  G = symmetrize(R + np.matmul(BtP, B))
  # make G positive definite so that smallest eigenvalue > delta.
  S, _ = np.linalg.eigh(G)
  G_ = G + np.maximum(0.0, delta - S[0]) * np.eye(G.shape[0])

  H = BtPA + M.T
  h = np.matmul(B.T, p) + np.matmul(BtP, c) + r

  K = -sp.linalg.solve(G_, H, sym_pos=True)
  k = -sp.linalg.solve(G_, h, sym_pos=True)

  H_GK = H + np.matmul(G, K)
  P = symmetrize(Q + AtPA + np.matmul(H_GK.T, K) + np.matmul(K.T, H))
  p = q + np.matmul(A.T, p) + np.matmul(AtP, c) + np.matmul(
      H_GK.T, k) + np.matmul(K.T, h)

  return P, p, K, k


@jit
def tvlqr(Q, q, R, r, M, A, B, c):
  """Discrete-time Finite Horizon Time-varying LQR.

  Note - for vectorization convenience, the leading dimension of R, r, M, A, B,
  C can be (T + 1) but the last row will be ignored.

  Args:
    Q: [T+1, n, n] numpy array.
    q: [T+1, n] numpy array.
    R: [T, m, m] numpy array.
    r: [T, m] numpy array.
    M: [T, n, m] numpy array.
    A: [T, n, n] numpy array.
    B: [T, n, m] numpy array.
    c: [T, n] numpy array.

  Returns:
    K: [T, m, n] Gains
    k: [T, m] Affine terms (u_t = np.matmul(K[t],  x_t) + k[t])
    P: [T+1, n, n] numpy array encoding initial value function.
    p: [T+1, n] numpy array encoding initial value function.
  """

  T = Q.shape[0] - 1
  m = R.shape[1]
  n = Q.shape[1]

  P = np.zeros((T+1, n, n))
  p = np.zeros((T+1, n))
  K = np.zeros((T, m, n))
  k = np.zeros((T, m))

  P = P.at[-1].set(Q[T])
  p = p.at[-1].set(q[T])

  def body(tt, inputs):
    K, k, P, p = inputs
    t = T - 1 - tt
    P_t, p_t, K_t, k_t = lqr_step(P[t+1], p[t+1], Q[t], q[t], R[t], r[t], M[t],
                                  A[t], B[t], c[t])
    K = K.at[t].set(K_t)
    k = k.at[t].set(k_t)
    P = P.at[t].set(P_t)
    p = p.at[t].set(p_t)

    return K, k, P, p

  return lax.fori_loop(0, T, body, (K, k, P, p))


@partial(jit, static_argnums=(0,))
def ctvlqr(projector, Q, q, R, r, M, A, B, c, x0, rho=1.0, maxiter=100):
  """Constrained Discrete-time Finite Horizon Time-varying LQR.

  Note - for vectorization convenience, the leading dimension of R, r, M, A, B,
  C can be (T + 1) but the last row will be ignored.

  Args:
    projector: X1, U1 = projector(X, U) projects X, U to the constraint set.
    Q: [T+1, n, n] numpy array.
    q: [T+1, n] numpy array.
    R: [T, m, m] numpy array.
    r: [T, m] numpy array.
    M: [T, n, m] numpy array.
    A: [T, n, n] numpy array.
    B: [T, n, m] numpy array.
    c: [T, n] numpy array.
    x0: [n] initial condition.
    rho: ADMM rho parameter.
    maxiter: maximum iterations.

  Returns:
    X: [T+1, n] state trajectory.
    U: [T, m] control sequencee.
    K: [T, m, n] Gains
    k: [T, m] Affine terms (u_t = np.matmul(K[t],  x_t) + k[t])

  Note: this implementation is ADMM-based and follows from section 5.2 of
        Distributed Optimization and Statistical Learning via the
        Alternating Direction Method of Multipliers, Boyd et.al. 2010.
        https://stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf
  """

  T, m, _ = R.shape
  n = Q.shape[1]
  X = np.zeros((T+1, n))
  U = np.zeros((T, m))
  VX = np.zeros((T+1, n))
  VU = np.zeros((T, m))
  ZX = np.zeros((T+1, n))
  ZU = np.zeros((T, m))
  K = np.zeros((T, m, n))
  k = np.zeros((T, m))
  Im = np.array([np.eye(m)]*T)
  In = np.array([np.eye(n)]*(T+1))

  def body(tt, inputs):
    del tt
    X, U, VX, VU, ZX, ZU, K, k = inputs
    K, k, _, _ = tvlqr(Q + rho*In,
                       q - (ZX - VX),
                       R + rho*Im,
                       r - (ZU - VU),
                       M, A, B, c)
    X, U = rollout(K, k, x0, A, B, c)
    ZX, ZU = projector(X + VX, U + VU)
    VX = VX + X - ZX
    VU = VU + U - ZU
    return (X, U, VX, VU, ZX, ZU, K, k)

  X, U, _, _, _, _, K, k = lax.fori_loop(0,
                                         maxiter, body,
                                         (X, U, VX, VU, ZX, ZU, K, k))
  return X, U, K, k
