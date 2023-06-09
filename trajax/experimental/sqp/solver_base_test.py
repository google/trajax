# pylint: disable=invalid-name
"""Tests for SQP solver base class."""

from absl.testing import absltest
import jax
from jax import random
from jax.config import config  # pylint: disable=g-importing-member
import jax.numpy as jnp
from trajax import integrators
from trajax import optimizers
from trajax.experimental.sqp import solver_base
from trajax.experimental.sqp import util


config.update('jax_enable_x64', True)


@jax.jit
def acrobot(x, u, t, params=(1.0, 1.0, 1.0, 0.5, 0.5, 1.0, 1.0)):
  """Classic Acrobot system.

  Note this implementation emulates the OpenAI gym implementation of
  Acrobot-v2, which itself is based on Stutton's Reinforcement Learning book.

  Args:
    x: state, (4, ) array
    u: control, (1, ) array
    t: scalar time. Disregarded because system is time-invariant.
    params: tuple of (LINK_MASS_1, LINK_MASS_2, LINK_LENGTH_1, LINK_COM_POS_1,
      LINK_COM_POS_2 LINK_MOI_1, LINK_MOI_2)

  Returns:
    xdot: state time derivative, (4, )
  """
  del t  # Unused

  m1, m2, l1, lc1, lc2, I1, I2 = params
  g = 9.8
  a = u[0]
  theta1 = x[0]
  theta2 = x[1]
  dtheta1 = x[2]
  dtheta2 = x[3]
  d1 = (
      m1 * lc1**2
      + m2 * (l1**2 + lc2**2 + 2 * l1 * lc2 * jnp.cos(theta2))
      + I1
      + I2
  )
  d2 = m2 * (lc2**2 + l1 * lc2 * jnp.cos(theta2)) + I2
  phi2 = m2 * lc2 * g * jnp.cos(theta1 + theta2 - jnp.pi / 2.0)
  phi1 = (
      -m2 * l1 * lc2 * dtheta2**2 * jnp.sin(theta2)
      - 2 * m2 * l1 * lc2 * dtheta2 * dtheta1 * jnp.sin(theta2)
      + (m1 * lc1 + m2 * l1) * g * jnp.cos(theta1 - jnp.pi / 2)
      + phi2
  )
  ddtheta2 = (
      a + d2 / d1 * phi1 - m2 * l1 * lc2 * dtheta1**2 * jnp.sin(theta2) - phi2
  ) / (m2 * lc2**2 + I2 - d2**2 / d1)
  ddtheta1 = -(d2 * ddtheta2 + phi1) / d1
  return jnp.array([dtheta1, dtheta2, ddtheta1, ddtheta2])


class TrajprobTest(absltest.TestCase):

  def setUp(self):
    """Setup a problem."""

    super().setUp()

    dynamics = integrators.euler(acrobot, dt=0.05)
    n, m, T = (4, 1, 20)
    self.n, self.m, self.T = (n, m, T)

    s1_indices = (0, 1)
    state_wrap = util.get_s1_wrapper(s1_indices)

    goal = jnp.array([jnp.pi, 0., 0., 0.])

    @jax.jit
    def cost(x, u, t, weights=(1., 1., 1.)):
      delta = state_wrap(x - goal)
      goal_cost = 2. + jnp.cos(x[0]) + jnp.cos(x[0] + x[1])
      stage_cost = 0.5 * (weights[0] * goal_cost + weights[1] * (u[0]**2.))
      term_cost = 0.5 * weights[2] * jnp.vdot(delta, delta)
      return jnp.where(t == T, term_cost, stage_cost)

    @jax.jit
    def state_constraint(x, t):
      # Require c_x(x[t], t) >= 0
      delta = state_wrap(x - goal)
      c_goal = jnp.where(t == T, 0.2**2. - jnp.vdot(delta, delta), 1.0)
      return jnp.array([c_goal])

    control_bounds = (jnp.array([-2.]), jnp.array([2.]))

    self.prob = solver_base.TrajectoryOptimizationSolver(
        n, m, T, dynamics, cost, control_bounds, state_constraint,
        s1_indices)

    self.assertEqual(self.prob._n_cx, 1)  # goal constraint only
    self.assertEqual(self.prob._n_cu, 2)

    # Setup test-specific common inputs
    self.x0 = jnp.zeros(self.n)
    self.U = random.uniform(random.PRNGKey(42), (self.T, self.m))
    self.X = self.prob._vec_wrap(optimizers.rollout(self.prob._dynamics,
                                                    self.U, self.x0))

    cost_weights = (0.1, 0.01, 10.)
    self.prob_params = ((cost_weights,), ())

  def testGetQPParams(self):
    """Test getting QP Params."""
    dyn_params, cost_grads, constraint_vals = self.prob._get_QP_params(
        self.U, self.X, self.prob_params)
    self.assertEqual(dyn_params[0].shape, (self.T, self.n, self.n))
    self.assertEqual(dyn_params[1].shape, (self.T, self.n, self.m))
    self.assertEqual(cost_grads[0].shape, (self.T + 1, self.n))
    self.assertEqual(cost_grads[1].shape, (self.T, self.m))
    self.assertEqual(constraint_vals[0].shape, (self.T, self.prob._n_cu))
    self.assertEqual(constraint_vals[2].shape, (self.T + 1, self.prob._n_cx))
    self.assertEqual(constraint_vals[1].shape,
                     (self.T, self.prob._n_cu, self.m))
    self.assertEqual(constraint_vals[3].shape,
                     (self.T + 1, self.prob._n_cx, self.n))

  def testAdjointHess(self):
    """Test Adjoint and Hessian computation."""
    dyn_params, cost_grads, constraint_vals = self.prob._get_QP_params(
        self.U, self.X, self.prob_params)
    Yu = jnp.zeros((self.T, self.prob._n_cu))
    Yx = jnp.zeros((self.T + 1, self.prob._n_cx))
    grad_u, adjoints = self.prob._compute_adjoint(
        (self.U, self.X), (Yu, Yx), dyn_params, cost_grads, constraint_vals)
    hess_params = self.prob._get_QP_Hess(
        self.U, self.X, Yu, Yx, adjoints, False, self.prob_params)

    self.assertEqual(grad_u.shape, (self.T, self.m))
    self.assertEqual(hess_params[0].shape,
                     (self.T, self.n + self.m, self.n + self.m))
    self.assertEqual(hess_params[1].shape,
                     (self.T + 1, self.n, self.n))

  def testMerit(self):
    """Test computing merit function."""
    Yu = jnp.zeros((self.T, self.prob._n_cu))
    Yx = jnp.zeros((self.T + 1, self.prob._n_cx))
    Su = jnp.zeros_like(Yu)
    Sx = jnp.zeros_like(Yx)
    Rho = random.uniform(random.PRNGKey(44), (self.T + 1,))
    merit = self.prob._merit_fn(self.U, self.X, Yu, Yx, Su, Sx, Rho,
                                self.prob_params)
    self.assertIsNot(merit, jnp.nan)
    self.assertIsNot(merit, jnp.inf)

  def testLQRCompute(self):
    """Test computing LQR gains."""
    dyn_params, _, _ = self.prob._get_QP_params(self.U, self.X,
                                                self.prob_params)
    gains = self.prob._get_LQR_control_gains(self.U, self.X, dyn_params, 1.,
                                             self.prob_params[0])
    self.assertEqual(gains.shape, (self.T, self.m, self.n))


if __name__ == '__main__':
  absltest.main()
