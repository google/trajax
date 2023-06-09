# pylint: disable=invalid-name
"""Tests for shootsqp."""

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax.config import config  # pylint: disable=g-importing-member
import jax.numpy as jnp
from trajax import integrators

from trajax.experimental.sqp import shootsqp
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


def _setupAcrobot(options_overwrite):
  """Setup acrobot problem and stabilized shooting-SQP solver."""
  dynamics = integrators.euler(acrobot, dt=0.05)
  n, m, T = (4, 1, 150)
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

  solver_options = dict(
      method=shootsqp.SQP_METHOD.STABLE,
      hess='gn',
      proj_init=True,
      verbose=False,
      max_iter=100,
      ls_eta=0.49,
      ls_beta=0.8,
      primal_tol=1e-3,
      dual_tol=1e-3,
      debug=False)
  solver_options.update(options_overwrite)
  solver = shootsqp.ShootSQP(n, m, T, dynamics, cost, control_bounds,
                             state_constraint, s1_indices, **solver_options)

  # Problem settings
  cost_weights = (0.1, 0.01, 10.)
  x0 = jnp.zeros((n,))
  U0 = jnp.zeros((T, m))
  X0 = jnp.linspace(x0, goal, T + 1)

  return solver, x0, U0, X0, ((cost_weights,), ())


def _setupCar4D(options_overwrite):
  """Setup 4d-car problem and open-loop shooting-SQP solver."""

  def car_ode(x, u, t):
    del t
    return jnp.array([x[3] * jnp.sin(x[2]),
                      x[3] * jnp.cos(x[2]),
                      x[3] * u[0],
                      u[1]])
  dt = 0.05
  dynamics = integrators.euler(car_ode, dt=dt)
  n, m, T = (4, 2, 40)
  s1_indices = (2,)
  state_wrap = util.get_s1_wrapper(s1_indices)

  R = jnp.diag(jnp.array([0.2, 0.1]))
  Q_T = jnp.diag(jnp.array([50., 50., 50., 10.]))
  goal_default = jnp.array([3., 3., jnp.pi/2, 0.])

  @jax.jit
  def cost(x, u, t, goal=goal_default):
    stage_cost = dt * jnp.vdot(u, R @ u)
    delta = state_wrap(x - goal)
    term_cost = jnp.vdot(delta, Q_T @ delta)
    return jnp.where(t == T, term_cost, stage_cost)

  control_bounds = (jnp.array([-jnp.pi/3., -6.]),
                    jnp.array([jnp.pi/3., 6.]))

  # Setup obstacle environment for state constraint
  obs = [(jnp.array([1., 1.]), 0.5),
         (jnp.array([1, 2.5]), 0.5),
         (jnp.array([2.5, 2.5]), 0.5)]
  def obs_constraint(pos):
    def avoid_obs(pos_c, ob):
      delta_body = pos_c - ob[0]
      delta_dist_sq = jnp.vdot(delta_body, delta_body) - (ob[1]**2)
      return delta_dist_sq
    return jnp.array([avoid_obs(pos, ob) for ob in obs])

  @jax.jit
  def state_constraint(x, t):
    del t
    pos = x[0:2]
    return obs_constraint(pos)

  solver_options = dict(
      method=shootsqp.SQP_METHOD.OPEN,
      hess='full',
      proj_init=False,
      verbose=False,
      max_iter=100,
      ls_eta=0.49,
      ls_beta=0.8,
      primal_tol=1e-3,
      dual_tol=1e-3,
      debug=False)
  solver_options.update(options_overwrite)
  solver = shootsqp.ShootSQP(n, m, T, dynamics, cost, control_bounds,
                             state_constraint, s1_indices, **solver_options)

  # Problem settings
  x0 = jnp.array([0.25, 1.75, 0., 0.])
  U0 = jnp.zeros((T, m))
  X0 = None

  return solver, x0, U0, X0, ((), ())


def _setupPoint2DWithLinearizedObstacleAvoidance(options_overwrite):
  """Setup 2d-point-mass problem and open-loop shooting-SQP solver."""

  dt = 0.1
  def dynamics(x, u, t, dt=dt):
    del t
    x_next = x + dt*u
    return x_next

  n, m, T = (2, 2, 40)

  R = jnp.eye(m)
  Q_T = jnp.eye(n)
  goal_default = 2*jnp.ones(n) + jnp.array([1.0, 0.0])

  @jax.jit
  def cost(x, u, t, goal=goal_default):
    stage_cost = dt * jnp.vdot(u, R @ u)
    delta = x-goal
    term_cost = jnp.vdot(delta, Q_T @ delta)
    return jnp.where(t == T, term_cost, stage_cost)

  control_bounds = (jnp.array([-1., -1.]), jnp.array([1., 1.]))

  # Setup obstacle environment for state constraint
  obs = [(jnp.array([1., 1.]), 0.5),
         (jnp.array([1, 2.5]), 0.5),
         (jnp.array([2.5, 2.5]), 0.5)]

  # The obstacle avoidance constraints are linearized, with linearization
  # coefficients that are updated at each iteration of the SQP solver.
  def state_constraint(x, t,
                       Xp=jnp.zeros((T+1, n)),
                       dists_to_obs=jnp.zeros((T+1, 3)),
                       dists_to_obs_dx=jnp.zeros((T+1, 3, n))):
    dists_linearized = dists_to_obs[t] + dists_to_obs_dx[t] @ (x - Xp[t])
    return dists_linearized

  state_constraints_params = (jnp.zeros((T+1, n)),
                              jnp.zeros((T+1, 3)),
                              jnp.zeros((T+1, 3, n)))
  def update_params(params, U, X):
    _, Xp = U, X
    # define
    @jax.jit
    def obs_constraint(pos):
      def avoid_obs(pos_c, ob):
        delta_body = pos_c - ob[0]
        delta_dist_sq = jnp.linalg.norm(delta_body) - ob[1]
        return delta_dist_sq
      return jnp.array([avoid_obs(pos, ob) for ob in obs])
    obs_constraint_dx = jax.jacobian(obs_constraint)
    # evaluate
    dists_to_obs = jax.vmap(obs_constraint)(Xp)
    dists_to_obs_dx = jax.vmap(obs_constraint_dx)(Xp)
    # pack parameters
    params = (params[0], (Xp, dists_to_obs, dists_to_obs_dx))
    return params

  solver_options = dict(
      method=shootsqp.SQP_METHOD.OPEN,
      hess='full',
      proj_init=False,
      verbose=False,
      max_iter=100,
      ls_eta=0.49,
      ls_beta=0.8,
      primal_tol=1e-2,
      dual_tol=1e-2,
      debug=False)
  solver_options.update(options_overwrite)
  solver = shootsqp.ShootSQP(n, m, T, dynamics, cost, control_bounds,
                             state_constraint,
                             update_params=update_params,
                             **solver_options)

  # Problem settings
  x0 = jnp.zeros((n,))
  U0 = jnp.zeros((T, m))
  X0 = None

  return solver, x0, U0, X0, ((), state_constraints_params)


_Methods = shootsqp.SQP_METHOD
_QP_solvers = shootsqp.QP_SOLVER
_Status = shootsqp.Status


class ShootsqpTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('acrobot-stable', 'acrobot', _Status.SOLVED, {}, 1e-3),
      ('car4d-open', 'car4d', _Status.MAXITER, {}, 1e-1),
      ('car4d-stable', 'car4d', _Status.SOLVED,
       {'method': _Methods.STABLE}, 1e-3),
      ('acrobot-stable-lqr', 'acrobot', _Status.SOLVED,
       {'qp_solver': _QP_solvers.QP_ALILQR}, 1e-3),
      ('car4d-stable-lqr', 'car4d', _Status.SOLVED,
       {'qp_solver': _QP_solvers.QP_ALILQR,
        'method': _Methods.STABLE}, 1e-3),
      ('point2d-open', 'point2d', _Status.SOLVED,
       {'qp_solver': _QP_solvers.QP_ALILQR}, 1e-3),
      ('acrobot-ddp', 'acrobot', _Status.SOLVED,
       {'method': _Methods.SENS,
        'ddp_options': {'ddp_gamma': 1e-4}}, 1e-3),
      ('car4d-ddp', 'car4d', _Status.SOLVED,
       {'method': _Methods.SENS, 'ddp_options': {'ddp_gamma': 1e-4}}, 1e-3),
      ('car4d-addp', 'car4d', _Status.SOLVED,
       {'method': _Methods.APPROX_SENS,
        'ddp_options': {'ddp_gamma': 1e-4, 'ddp_maxiter': 1}}, 1e-3),
      ('acrobot-addp', 'acrobot', _Status.SOLVED,
       {'method': _Methods.APPROX_SENS,
        'ddp_options': {'ddp_gamma': 1e-4, 'ddp_maxiter': 1}}, 1e-3))

  def testSolve(self, system_name, expected_status, options_overwrite, tol):
    """Sets up and solves a trajectory optimization problem."""

    if system_name == 'acrobot':
      solver, x0, U0, X0, prob_params = _setupAcrobot(options_overwrite)
    elif system_name == 'car4d':
      solver, x0, U0, X0, prob_params = _setupCar4D(options_overwrite)
    elif system_name == 'point2d':
      outputs = _setupPoint2DWithLinearizedObstacleAvoidance(options_overwrite)
      solver, x0, U0, X0, prob_params = outputs
    else:
      raise ValueError(f'System {system_name} not recognized.')

    # Solve problem
    solution = solver.solve(x0, U0, X0, params=prob_params)
    itr = solution.iterations
    history = solution.history
    primals = solution.primals
    duals = solution.duals
    kkt_resid = solution.kkt_residuals

    # Print result
    logging.info('%s; status: %s, itr %d, obj %.2f',
                 system_name, solution.status.name, itr, solution.objective)

    # Assert sizes of results as expected
    self.assertLessEqual(itr, solver.opt.max_iter)
    for arr in history.values():
      self.assertLen(arr, itr)
    self.assertLen(solution.times, itr)

    self.assertEqual(primals[0].shape, (solver._T, solver._m))
    self.assertEqual(primals[1].shape, (solver._T + 1, solver._n))
    self.assertEqual(duals[0].shape, (solver._T, solver._n_cu))
    self.assertEqual(duals[1].shape, (solver._T + 1, solver._n_cx))

    # Assert result status as expected
    self.assertEqual(history['obj'][-1], solution.objective)
    self.assertEqual(solution.status, expected_status)

    # Check solution quality
    self.assertGreaterEqual(min(kkt_resid['primal']), -tol)
    self.assertGreaterEqual(min(kkt_resid['dual']), -tol)
    self.assertLessEqual(max(kkt_resid['cslack']), tol)

    # If using SENS - check DDP computation correct:
    if solver.opt.method == _Methods.SENS:
      self.assertLessEqual(max(history['ddp_err']), 0.15)
      self.assertLessEqual(max(history['ddp_err_grad']), 1e-3)
    elif solver.opt.method == _Methods.APPROX_SENS:
      self.assertLessEqual(max(history['ddp_err']), 0.1)


if __name__ == '__main__':
  absltest.main()
