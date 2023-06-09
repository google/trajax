# pylint: disable=invalid-name
"""Utilities for Shooting-SQP."""

import functools
from typing import Tuple, Any, Callable, Optional

import jax
from jax import lax
import jax.numpy as jnp

from trajax import optimizers

STATE = jnp.ndarray
CONTROL = jnp.ndarray


def _wrap_to_pi(x: jnp.ndarray) -> jnp.ndarray:
  """Wraps x to lie within [-pi, pi]."""
  return (x + jnp.pi) % (2 * jnp.pi) - jnp.pi


def get_s1_wrapper(
    s1_ind: Optional[Tuple[int, ...]]) -> Callable[[STATE], STATE]:
  """Returns a function for wrapping S1 components of state to [-pi, pi]."""

  id_wrap = lambda x: x

  if s1_ind is not None:
    idxs = jnp.array(s1_ind)
    def state_wrapper(x: STATE) -> STATE:
      x = x.at[idxs].set(_wrap_to_pi(x[idxs]))
      return x
    return jax.jit(state_wrapper)
  else:
    return jax.jit(id_wrap)


class Rollouts(object):
  """Class collecting different rollout functions used by SQP.

  NOTE: This class is initialized within an SQP solver to define internal
  rollout routines.
  """

  def __init__(self, dynamics: Callable[[STATE, CONTROL, int], STATE],
               state_wrap, vec_wrap,
               control_bounds: Tuple[CONTROL, CONTROL], T: int):
    """Initialize class.

    Args:
      dynamics: Callable, (x:(n,) array, u:(m,) array, t:int) --> (n,) array.
      state_wrap: Callable returned by get_s1_wrapper.
      vec_wrap: vmapped version of state_wrap.
      control_bounds: tuple of (u_lower: (m,) ndarray, u_upper: (m,) ndarray),
          defining bounds on control; use jnp.inf for unconstrained components.
      T: horizon (int, >= 2).
    """

    self._dynamics = dynamics
    self._state_wrap = state_wrap
    self._vec_wrap = vec_wrap
    self._control_bounds = control_bounds
    self._timesteps = jnp.arange(T)

    self.default_rollout = functools.partial(optimizers.rollout, self._dynamics)

  def _rollout(self, U, X, dU_d, dX_d, alpha, K=None):
    """Template rollout function.

    Args:
      U: control trajectory, ndarray: [T, m]
      X: state trajectory, ndarray: [T+1, n]
      dU_d: delta control trajectory from QP sub-problem, ndarray: [T, m]
      dX_d: delta state trajectory from QP sub-problem, ndarray: [T+1, n]
      alpha: float, step-size
      K: rollout gains (for closed-loop rollout only), ndarray: [T, m, n]

    Returns:
      U_al: updated control trajectory, ndarray: [T, m]
      X_al: updated state trajectory, ndarray: [T+1, n]
    """
    raise NotImplementedError

  def get_open_rollout(self):
    """Create open-loop rollout function."""
    roller = self.default_rollout

    def rollout(U, X, dU_d, dX_d, alpha):
      del dX_d
      X_al = self._vec_wrap(roller(U + alpha * dU_d, X[0]))
      return U + alpha * dU_d, X_al

    return jax.jit(rollout)

  def get_closed_rollout(self):
    """Create closed-loop tracking rollout function."""

    def roller(X, U, dX_d, dU_d, alpha, K):
      dx_0 = jnp.zeros_like(dX_d[0])

      def dynamics_for_scan(dx_k, k):
        diff = self._state_wrap(dx_k - alpha * dX_d[k])
        u_hat = U[k] + alpha * dU_d[k] + K[k] @ diff
        du_k = jnp.clip(u_hat,
                        self._control_bounds[0], self._control_bounds[1]) - U[k]
        x_ = self._dynamics(X[k] + self._state_wrap(dx_k), U[k] + du_k, k)
        dx_ = x_ - X[k+1]
        return dx_, (dx_k, du_k)

      dx_T_al, roll_out = lax.scan(dynamics_for_scan, dx_0, self._timesteps)
      dX_al, dU_al = roll_out
      dX_al = jnp.vstack((dX_al, dx_T_al))
      return dX_al, dU_al

    def rollout(U, X, dU_d, dX_d, alpha, K):
      dX_al, dU_al = roller(X, U, dX_d, dU_d, alpha, K)
      X_al = self._vec_wrap(X + dX_al)

      return U + dU_al, X_al

    return jax.jit(rollout)

  def get_proj_rollout(self):
    """Create projected-initialization rollout function."""

    rollout = self.get_closed_rollout()

    def proj_init(U0, X0, K):
      # implement: u_k = clip(U0_k + K_k @ (x_k - X0_k))
      return rollout(U0, X0, jnp.zeros_like(U0), jnp.zeros_like(X0), 0., K)

    return jax.jit(proj_init)


@jax.jit
def safe_cubic_opt(x1: float, x2: float,
                   vg1: Tuple[float, float], vg2: Tuple[float, float]):
  """Safe cubic optimization between x1 and x2.

  Reference: http://www-personal.umich.edu/~murty/611/611slides9.pdf

  Args:
    x1: left endpoint
    x2: right endpoint, larger than x1.
    vg1: tuple of (value, derivative) of real-valued function at left endpoint.
    vg2: tuple of (value, derivative) of real-valued function at right endpoint.

  Returns:
    armin(f(x)) for x in [x1, x2], where f is the cubic interpolant of function.
  """
  v1, g1 = vg1
  v2, g2 = vg2

  b1 = g1 + g2 - 3. * ((v1 - v2) / (x1 - x2))
  b2 = jnp.sqrt(jnp.maximum(b1 * b1 - g1 * g2, 0.))
  x = x2 - (x2 - x1) * ((g2 + b2 - b1) / (g2 - g1 + 2. * b2))
  return jnp.maximum(x1, jnp.minimum(x, x2))


def while_loop(cond_fun: Callable[[Any], Any],
               body_fun: Callable[[Any], Any],
               init_val: Any):
  """Generic while loop; same syntax as lax.while_loop."""
  val = init_val
  while cond_fun(val):
    val = body_fun(val)
  return val
