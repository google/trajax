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
"""Tests for trajax.optimizers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os

from absl.testing import absltest
from absl.testing import parameterized
from frozendict import frozendict
import jax
from jax import device_put
from jax import vmap
from jax.config import config
import jax.flatten_util
import jax.numpy as np
import numpy as onp
from six.moves import range
from trajax import optimizers
from trajax.integrators import euler
from trajax.integrators import rk4

config.update('jax_enable_x64', True)

# TODO(sindhwani): np.nan semantics currently requires the following flag.
os.environ['XLA_FLAGS'] = '--xla_cpu_enable_fast_math=false'


def grad_fd(fun):
  """Returns finite difference gradient evaluator of a scalar-valued function.

  Args:
    fun: function with signature y = fun(x) where x is (n, ) and y is a scalar.

  Returns:
    A jax.grad like function that evaluates numerical gradient of fun, i.e.,
          fd = grad_fd(fun)
          y = fd(x, h)
    then y is the finite difference gradient where,
              y[i] = [fun(x + h e_i) - fun(x)]/h
  """
  vfun = vmap(fun)

  def fd(x, h=1e-6):
    I = np.eye(x.shape[0])
    return (vfun(x + h * I) - fun(x)) / h

  return fd


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
      m1 * lc1**2 + m2 * (l1**2 + lc2**2 + 2 * l1 * lc2 * np.cos(theta2)) + I1 +
      I2)
  d2 = m2 * (lc2**2 + l1 * lc2 * np.cos(theta2)) + I2
  phi2 = m2 * lc2 * g * np.cos(theta1 + theta2 - np.pi / 2.)
  phi1 = (-m2 * l1 * lc2 * dtheta2**2 * np.sin(theta2) -
          2 * m2 * l1 * lc2 * dtheta2 * dtheta1 * np.sin(theta2) +
          (m1 * lc1 + m2 * l1) * g * np.cos(theta1 - np.pi / 2) + phi2)
  ddtheta2 = ((a + d2 / d1 * phi1 -
               m2 * l1 * lc2 * dtheta1**2 * np.sin(theta2) - phi2) /
              (m2 * lc2**2 + I2 - d2**2 / d1))
  ddtheta1 = -(d2 * ddtheta2 + phi1) / d1
  return np.array([dtheta1, dtheta2, ddtheta1, ddtheta2])


def pendulum(state, action, t, params=(1.0, 1.0, 9.81)):
  del t
  theta, theta_dot = state
  m, l, g = params
  return np.array([
      theta_dot,
      (np.squeeze(action) - m * g * l * np.sin(theta)) / (m * l * l)])


class OptimizersTest(parameterized.TestCase):

  def setUp(self):
    super(OptimizersTest, self).setUp()

    n, m, T = (8, 2, 16)
    H = onp.random.randn(T, n + m, n + m)
    self.H = device_put(H + onp.transpose(H, (0, 2, 1)))
    self.h = device_put(onp.random.randn(T, n + m))
    self.A = device_put(onp.random.randn(T, n, n))
    self.B = device_put(onp.random.randn(T, n, m))
    self.x0 = device_put(onp.random.randn(n))
    self.X = onp.random.randn(T, n)
    self.U = onp.random.randn(T, m)

    self.n = n
    self.m = m
    self.T = T

    def cost(x, u, t, params):
      z = np.concatenate([x, u])
      return params[0] * 0.5 * np.matmul(z.T, np.matmul(
          self.H[t], z)) + params[1] * np.dot(self.h[t], z)

    self.cost = cost

    def dynamics(x, u, t, params):
      return params[0] * np.matmul(self.A[t], x) + params[1] * np.matmul(
          self.B[t], u)

    self.dynamics = dynamics

  def testLinearize(self):
    n, m, T = (8, 2, 16)
    tol = 1e-5

    params = np.array([2.0, 3.0])
    linearize = optimizers.linearize(self.dynamics)
    Jx, Ju = linearize(self.X, self.U, np.arange(self.T), params)

    # Test linearization of dynamics without parameters.
    def dynamics_without_params(x, u, t):
      return self.dynamics(x, u, t, params)

    linearize = optimizers.linearize(dynamics_without_params)
    Jx1, Ju1 = linearize(self.X, self.U, np.arange(self.T))
    self.assertLess(np.linalg.norm(Jx1.flatten() - Jx1.flatten()), tol)
    self.assertLess(np.linalg.norm(Ju1.flatten() - Ju1.flatten()), tol)

    self.assertEqual(Jx.shape, (T, n, n))
    self.assertEqual(Ju.shape, (T, n, m))
    self.assertLess(
        np.linalg.norm(Jx.flatten() - params[0] * self.A.flatten()), tol)
    self.assertLess(
        np.linalg.norm(Ju.flatten() - params[1] * self.B.flatten()), tol)

    # Cost Linearization.
    linearize = optimizers.linearize(self.cost)
    q, r = linearize(self.X, self.U, np.arange(T), params)

    # Test linearization of cost without parameters.
    def cost_without_params(x, u, t):
      return self.cost(x, u, t, params)

    linearize = optimizers.linearize(cost_without_params)
    q1, r1 = linearize(self.X, self.U, np.arange(T))
    self.assertLess(np.linalg.norm(q - q1), tol)
    self.assertLess(np.linalg.norm(r - r1), tol)

    def grad(x, u, t):
      return params[0] * np.matmul(self.H[t], np.concatenate(
          (x, u))) + params[1] * self.h[t]

    for t in range(T):
      g = grad(self.X[t], self.U[t], t)
      self.assertLess(np.linalg.norm(q[t] - g[0:n]), tol)
      self.assertLess(np.linalg.norm(r[t] - g[n:]), tol)

  def testQuadratize(self):
    n, m, T = (self.n, self.m, self.T)
    tol = 1e-5

    H = self.H
    X, U = (self.X, self.U)
    params = np.array([2.0, 3.0])
    quadratize = optimizers.quadratize(self.cost)
    Q, R, M = quadratize(X, U, np.arange(T), params)

    #
    def cost_without_params(x, u, t):
      return self.cost(x, u, t, params)

    quadratize = optimizers.quadratize(cost_without_params)
    Q1, R1, M1 = quadratize(X, U, np.arange(T))
    self.assertLess(np.linalg.norm(Q1.flatten() - Q.flatten()), tol)
    self.assertLess(np.linalg.norm(R1.flatten() - R.flatten()), tol)
    self.assertLess(np.linalg.norm(M1.flatten() - M.flatten()), tol)

    self.assertEqual(Q.shape, (T, n, n))
    self.assertEqual(R.shape, (T, m, m))
    self.assertEqual(M.shape, (T, n, m))
    self.assertLess(
        np.linalg.norm(Q.flatten() - params[0] * H[:, 0:n, 0:n].flatten()), tol)
    self.assertLess(
        np.linalg.norm(R.flatten() - params[0] * H[:, n:, n:].flatten()), tol)
    self.assertLess(
        np.linalg.norm(M.flatten() - params[0] * H[:, 0:n, n:].flatten()), tol)

  def testEvaluate(self):
    n = 8
    m = 2
    T = 16
    params = 2.0
    cost = lambda x, u, t, params: params * 1.0
    evaluate = functools.partial(optimizers.evaluate, cost)
    X, U = onp.random.randn(T + 1, m), onp.random.randn(T, n)
    self.assertEqual(
        onp.sum(evaluate(X, optimizers.pad(U), params)), params * (T + 1))

  def testFiniteDifference(self):
    fun = lambda x: np.sum(x**2)
    x = np.array([1.0, 2.0, 3.0])
    tol = 1e-5
    self.assertLess(np.linalg.norm(grad_fd(fun)(x) - 2.0 * x), tol)

  def testRollout(self):
    T = 50
    roll = functools.partial(optimizers._rollout, rk4(acrobot, dt=0.1))
    x0 = 0.1 * onp.random.randn(4)
    U = 0.1 * onp.random.randn(T, 1)
    X = roll(U, x0)
    self.assertEqual(X.shape, (T + 1, 4))

  def testGradient(self):
    T = 50
    goal = np.array([np.pi, 0.0, 0.0, 0.0])
    dynamics = rk4(acrobot, dt=0.1)

    def cost(x, u, t, params):
      delta = x - goal
      terminal_cost = 0.5 * params[0] * np.dot(delta, delta)
      stagewise_cost = 0.5 * params[1] * np.dot(
          delta, delta) + 0.5 * params[2] * np.dot(u, u)
      return np.where(t == T, terminal_cost, stagewise_cost)

    x0 = 0.1 * onp.random.randn(4)
    U = 0.1 * onp.random.randn(T, 1)
    params_cost = (1000.0, 0.1, 0.01)

    objective = functools.partial(optimizers.objective,
                                  functools.partial(cost, params=params_cost),
                                  dynamics)
    grad = functools.partial(optimizers.grad_wrt_controls, cost, dynamics)
    gradient = grad(U, x0, (params_cost,), ())

    def obj(Uflat):
      return objective(np.reshape(Uflat, (T, 1)), x0)

    gradient_fd = grad_fd(obj)(U.flatten()).reshape((T, 1))
    tol = 1e-4
    self.assertLess(np.linalg.norm(gradient - gradient_fd), tol)

  def testCustomGradient(self):
    T = 50
    goal = np.array([np.pi, 0.0, 0.0, 0.0])
    dynamics = rk4(acrobot, dt=0.1)

    def cost(x, u, t, params):
      delta = x - goal
      terminal_cost = 0.5 * params[0] * np.dot(delta, delta)
      stagewise_cost = 0.5 * params[1] * np.dot(
          delta, delta) + 0.5 * params[2] * np.dot(u, u)
      return np.where(t == T, terminal_cost, stagewise_cost)

    x0 = 0.1 * onp.random.randn(4)
    U = 0.1 * onp.random.randn(T, 1)
    params_cost = (1000.0, 0.1, 0.01)

    obj = functools.partial(optimizers.objective,
                            functools.partial(cost, params=params_cost),
                            dynamics)
    grad = functools.partial(optimizers.grad_wrt_controls, cost, dynamics)
    gradient = grad(U, x0, (params_cost,), ())
    jax_gradient = jax.grad(obj)(U, x0)
    self.assertTrue(np.allclose(gradient, jax_gradient))

    def f(U, x0):
      return 2 * obj(U, x0)

    self.assertTrue(np.allclose(jax.grad(f)(U, x0), 2 * jax_gradient))

  def testAcrobotSolve(self):
    T = 50
    goal = np.array([np.pi, 0.0, 0.0, 0.0])
    dynamics = euler(acrobot, dt=0.1)

    def cost(x, u, t, params):
      delta = x - goal
      terminal_cost = 0.5 * params[0] * np.dot(delta, delta)
      stagewise_cost = 0.5 * params[1] * np.dot(
          delta, delta) + 0.5 * params[2] * np.dot(u, u)
      return np.where(t == T, terminal_cost, stagewise_cost)

    x0 = np.zeros(4)
    U = np.zeros((T, 1))
    params = np.array([1000.0, 0.1, 0.01])
    true_obj = 4959.476212
    self.assertLess(
        np.abs(
            optimizers.objective(
                functools.partial(cost, params=params), dynamics, U, x0) -
            true_obj), 1e-6)

    _, _, obj, gradient, _, _, _ = optimizers.ilqr(
        functools.partial(cost, params=params), dynamics, x0, U,
        maxiter=100, make_psd=False)
    optimal_obj = 51.0
    self.assertLess(obj, optimal_obj)
    self.assertLess(np.linalg.norm(gradient), 1e-4)

    _, _, obj, gradient, _ = optimizers.scipy_minimize(
        functools.partial(cost, params=params),
        dynamics,
        x0,
        U,
        method='Newton-CG',
        options={
            'xtol': 1e-10,
            'maxiter': 1000
        })
    self.assertLess(obj, optimal_obj)
    self.assertLess(np.linalg.norm(gradient), 1e-4)

  @parameterized.named_parameters(
      ('explicit', 'explicit', {}),
      ('cg', 'cg', {'tol': 1e-8}),
      ('tvlqr', 'tvlqr', {}),
      ('tvlqr_experimental', 'tvlqr_experimental', {}))
  def testCustomVJP(self, vjp_method, vjp_options):
    horizon = 5
    dynamics = rk4(pendulum, dt=0.01)

    # wrap to [-pi, pi]
    def angle_wrap(theta):
      return (theta + np.pi) % (2 * np.pi) - np.pi

    true_params = (10, 1, 0.01)
    def cost(params, state, action, t):
      final_weight, stage_weight, action_weight = params
      theta, theta_dot = state
      theta_err = angle_wrap(theta - np.pi)
      state_cost = stage_weight * (theta_err**2 + theta_dot**2)
      action_cost = action_weight * np.squeeze(action)**2
      return np.where(t == horizon, final_weight * state_cost,
                      state_cost + action_cost)

    key = jax.random.PRNGKey(0)
    n_trajectories = 3
    x0s = jax.random.normal(key, shape=(n_trajectories, 2))

    def solve(params, x0):
      u0 = np.zeros((horizon, 1))
      xs, us, *_ = optimizers.ilqr(
          functools.partial(cost, params), dynamics, x0, u0,
          vjp_method=vjp_method,
          vjp_options=vjp_options)
      return xs, us

    expert = functools.partial(solve, true_params)
    train_xs, train_us = jax.vmap(expert)(x0s)

    def train_loss(params):
      proposal = functools.partial(solve, params)
      proposed_xs, proposed_us = jax.vmap(proposal)(train_xs[:, 0])
      return (np.sum(np.square(train_xs - proposed_xs)) +
              np.sum(np.square(train_us - proposed_us)))

    grad_fn = jax.grad(train_loss)
    params = np.ones(3)
    grad = grad_fn(params)
    assert params.shape == grad.shape
    assert np.linalg.norm(grad_fn(np.array(true_params))) <= 1e-5

  @parameterized.named_parameters(
      ('explicit', 'explicit', {}),
      ('cg', 'cg', {'tol': 1e-8}),
      ('tvlqr', 'tvlqr', {}),
      ('tvlqr_experimental', 'tvlqr_experimental', {}))
  def testCustomVJPPyTree(self, vjp_method, vjp_options):
    horizon = 5
    dynamics = rk4(pendulum, dt=0.01)

    # wrap to [-pi, pi]
    def angle_wrap(theta):
      return (theta + np.pi) % (2 * np.pi) - np.pi

    def cost(params, state, action, t):
      final_weight, stage_weight, action_weight = params
      theta, theta_dot = state
      theta_err = angle_wrap(theta - np.pi)
      state_cost = stage_weight * (theta_err**2 + theta_dot**2)
      action_cost = action_weight * np.squeeze(action)**2
      return np.where(t == horizon, final_weight * state_cost,
                      state_cost + action_cost)

    def pytree_cost(params, state, action, t):
      final_weight, stage_weight, action_weight = (params['final_weight'],
                                                   params['stage_weight'],
                                                   params['action_weight'])
      return cost(
          np.array([final_weight, stage_weight, action_weight]), state, action,
          t)

    def multiparam_cost(final_weight, stage_weight, action_weight,
                        state, action, t):
      return cost(
          np.array([final_weight, stage_weight, action_weight]), state, action,
          t)

    key = jax.random.PRNGKey(0)
    x0 = jax.random.normal(key, shape=(2,))

    def loss(params):
      u0 = np.zeros((horizon, 1))
      xs, us, *_ = optimizers.ilqr(
          functools.partial(cost, params), dynamics, x0, u0,
          vjp_method=vjp_method,
          vjp_options=vjp_options)
      return np.sum(np.square(xs)) + np.sum(np.square(us))

    def pytree_loss(params):
      u0 = np.zeros((horizon, 1))
      xs, us, *_ = optimizers.ilqr(
          functools.partial(pytree_cost, params), dynamics, x0, u0,
          vjp_method=vjp_method,
          vjp_options=vjp_options)
      return np.sum(np.square(xs)) + np.sum(np.square(us))

    def multiparam_loss(final_weight, stage_weight, action_weight):
      u0 = np.zeros((horizon, 1))
      xs, us, *_ = optimizers.ilqr(
          functools.partial(multiparam_cost,
                            final_weight, stage_weight, action_weight),
          dynamics, x0, u0,
          vjp_method=vjp_method,
          vjp_options=vjp_options)
      return np.sum(np.square(xs)) + np.sum(np.square(us))

    params = np.array([10.0, 1.0, 0.01])
    pytree_params = {
        'final_weight': 10.0,
        'stage_weight': 1.0,
        'action_weight': 0.01
    }

    grad_params = jax.jacobian(loss)(params)
    grad_pytree_params = jax.jacobian(pytree_loss)(pytree_params)
    assert np.allclose(
        grad_params,
        np.array([grad_pytree_params['final_weight'],
                  grad_pytree_params['stage_weight'],
                  grad_pytree_params['action_weight']]))

    for idx, grad_param in enumerate(grad_params):
      assert np.allclose(
          grad_param,
          jax.jacobian(multiparam_loss, argnums=idx)(*params))

  def testPendulumReadmeExample(self):
    horizon = 300
    dynamics = rk4(pendulum, dt=0.01)

    # wrap to [-pi, pi]
    def angle_wrap(theta):
      return (theta + np.pi) % (2 * np.pi) - np.pi

    true_params = (100.0, 10.0, 1.0)
    def cost(params, state, action, t):
      final_weight, stage_weight, action_weight = params
      theta, theta_dot = state
      theta_err = angle_wrap(theta - np.pi)
      state_cost = stage_weight * (theta_err**2 + theta_dot**2)
      action_cost = action_weight * np.squeeze(action)**2
      return np.where(t == horizon, final_weight * state_cost,
                      state_cost + action_cost)
    x0 = np.array([-0.9, 0])
    U0 = np.zeros((horizon, 1))
    X, _, _, _, _, _, _ = optimizers.ilqr(
        functools.partial(cost, true_params), dynamics, x0, U0)
    assert np.abs(angle_wrap(X[-1, 0] - np.pi)) <= 0.1
    assert np.abs(X[-1, 1]) <= 0.1

  def testCustomVJPConsistency(self):
    vjp_methods = ('explicit', 'cg', 'tvlqr', 'tvlqr_experimental')
    vjp_options_per_method = ({}, {'tol': 1e-8}, {})

    T = 5
    goal = np.array([np.pi, 0.0, 0.0, 0.0])
    dynamics = euler(acrobot, dt=0.1)

    def cost(x, u, t, params):
      delta = x - goal
      terminal_cost = 0.5 * params[0] * np.dot(delta, delta)
      stagewise_cost = 0.5 * params[1] * np.dot(
          delta, delta) + 0.5 * params[2] * np.dot(u, u)
      return np.where(t == T, terminal_cost, stagewise_cost)

    x0 = np.zeros(4)
    U0 = np.zeros((T, 1))
    params = np.array([100.0, 0.1, 0.5])

    def make_jac_function(vjp_method, vjp_options):
      def fn(params):
        X, U, _, _, _, _, _ = optimizers.ilqr(
            lambda x, u, t: cost(x, u, t, params),
            dynamics,
            x0,
            U0,
            vjp_method=vjp_method,
            vjp_options=vjp_options)
        return X, U
      return jax.jacobian(fn)

    jacs = [make_jac_function(vjp_method, vjp_options)(params)
            for vjp_method, vjp_options in
            zip(vjp_methods, vjp_options_per_method)]
    def pytree_allclose(a, b, *args, **kwargs):
      return jax.flatten_util.ravel_pytree(
          jax.tree_map(lambda x, y: np.allclose(x, y, *args, **kwargs), a,
                       b))[0].all()
    for dX, dU in jacs[1:]:
      self.assertTrue(pytree_allclose(jacs[0][0], dX, atol=1e-6, rtol=1e-6))
      self.assertTrue(pytree_allclose(jacs[0][1], dU, atol=1e-6, rtol=1e-6))

  @parameterized.named_parameters(
      ('explicit', 'explicit', {}),
      ('cg', 'cg', {'tol': 1e-8}),
      ('tvlqr_experimental', 'tvlqr_experimental', {}))
  def testGradientOptimization(self, vjp_method, vjp_options):
    def cost(x, u, t, Q, R):
      del t
      return np.dot(x, Q @ x) + np.dot(u, R @ u)

    def dynamics(x, u, t, A, B):
      del t
      return A @ x + B @ u

    def random_symmetric_matrix(key, dim, min_eval, max_eval):
      k1, k2 = jax.random.split(key)
      W = jax.random.normal(k1, shape=(dim, dim))
      Q, _ = np.linalg.qr(W)
      evals = jax.random.uniform(
          k2, shape=(dim,), minval=min_eval, maxval=max_eval)
      return Q @ np.diag(evals) @ Q.T

    n, d, T = 5, 3, 10

    Q_true = random_symmetric_matrix(jax.random.PRNGKey(0), n, 1.0, 10.0)
    R_true = random_symmetric_matrix(jax.random.PRNGKey(1), d, 1.0, 10.0)
    A_true = jax.random.normal(jax.random.PRNGKey(2), shape=(n, n))
    B_true = jax.random.normal(jax.random.PRNGKey(3), shape=(n, d))
    x0_true = jax.random.normal(jax.random.PRNGKey(4), shape=(n,))

    Q_perturb = random_symmetric_matrix(jax.random.PRNGKey(5), n, -0.5, 0.5)
    R_perturb = random_symmetric_matrix(jax.random.PRNGKey(6), d, -0.5, 0.5)
    A_perturb = 0.05 * jax.random.normal(jax.random.PRNGKey(7), shape=(n, n))
    B_perturb = 0.05 * jax.random.normal(jax.random.PRNGKey(8), shape=(n, d))
    x0_perturb = 0.1 * jax.random.normal(jax.random.PRNGKey(9), shape=(n,))

    @functools.partial(jax.jit, static_argnums=(5,))
    def make_optimal_traj(A, B, Q, R, x0, T):
      X, U, _, _, _, _, _ = optimizers.ilqr(
          functools.partial(cost, Q=Q, R=R),
          functools.partial(dynamics, A=A, B=B),
          x0,
          np.zeros((T, B.shape[1])))
      return X, U
    X_true, U_true = make_optimal_traj(
        A_true, B_true, Q_true, R_true, x0_true, T)

    def loss(A, B, Q, R, x0, X, U):
      X_ilqr, U_ilqr, _, _, _, _, _ = optimizers.ilqr(
          functools.partial(cost, Q=Q, R=R),
          functools.partial(dynamics, A=A, B=B),
          x0,
          np.zeros((T, B.shape[1])),
          vjp_method=vjp_method,
          vjp_options=vjp_options)
      return np.sum(np.square(X_ilqr - X)) + np.sum(np.square(U_ilqr - U))

    @functools.partial(jax.jit, static_argnums=(1, 9))
    def opt_params(param_init, param_index, A_true, B_true, Q_true, R_true,
                   x0_true, X_true, U_true, num_iters, step_size):
      assert param_index >= 0 and param_index <= 4
      grad_param = jax.grad(loss, argnums=param_index)
      true_params = (A_true, B_true, Q_true, R_true, x0_true, X_true, U_true)

      def get_args(param):
        return (true_params[:param_index] + (param,) +
                true_params[param_index + 1:])

      def scan_fn(param, _):
        param_next = param - step_size * grad_param(*get_args(param))
        return param_next, loss(*get_args(param_next))

      loss_init = loss(*get_args(param_init))
      param_final, losses = jax.lax.scan(scan_fn, param_init,
                                         xs=np.arange(num_iters))
      return param_final, np.hstack((loss_init, losses))

    params_true = (A_true, B_true, Q_true, R_true, x0_true, X_true, U_true)
    arguments = (
        (A_true + A_perturb, 0) + params_true + (5000, 0.001),
        (B_true + B_perturb, 1) + params_true + (5000, 0.001),
        (Q_true + Q_perturb, 2) + params_true + (5000, 0.01),
        (R_true + R_perturb, 3) + params_true + (5000, 0.1),
        (x0_true + x0_perturb, 4) + params_true + (5000, 0.001),
    )

    for args in arguments:
      _, losses = opt_params(*args)
      self.assertLessEqual(losses[-1], 0.01)

  def testCEMUpdateMeanStdev(self):
    num_samples, horizon, dim_control = 400, 20, 1
    old_mean, old_stdev = np.zeros((horizon, dim_control)), np.ones(
        (horizon, dim_control))
    sampled_controls = np.concatenate(
        (np.zeros((num_samples // 2, horizon, dim_control)),
         np.ones((num_samples // 2, horizon, dim_control))),
        axis=0)
    # Construct costs such that control 1 has cost 0, and control 0 has cost 1
    costs = np.hstack((np.ones(num_samples // 2), np.zeros(num_samples // 2)))
    hyperparams = frozendict({
        'num_samples': num_samples,
        'elite_portion': 0.1,
        'evolution_smoothing': 0.
    })
    updated_mean, updated_stdev = optimizers.cem_update_mean_stdev(
        old_mean, old_stdev, sampled_controls, costs, hyperparams)
    self.assertEqual(updated_mean.shape, (horizon, dim_control))
    self.assertEqual(updated_stdev.shape, (horizon, dim_control))
    self.assertTrue(np.allclose(updated_mean, np.ones_like(updated_mean)))
    self.assertTrue(np.allclose(updated_stdev, np.zeros_like(updated_stdev)))

  def testConstrainedAcrobotSolve(self):
    T = 50
    goal = np.array([np.pi, 0.0, 0.0, 0.0])
    u_lower = -5.0  # control limits
    u_upper = 5.0  # control limits
    dynamics = euler(acrobot, dt=0.1)

    def cost(x, u, t, params):
      delta = x - goal
      terminal_cost = 0.5 * params[0] * np.dot(delta, delta)
      stagewise_cost = 0.5 * params[1] * np.dot(
          delta, delta) + 0.5 * params[2] * np.dot(u, u)
      return np.where(t == T, terminal_cost, stagewise_cost)

    def equality_constraint(x, u, t):
      del u

      # maximum constraint dimension across time steps
      dim = 4

      def goal_constraint(x):
        err = x - goal
        return err

      return np.where(t == T, goal_constraint(x), np.zeros(dim))

    def inequality_constraint(x, u, t):
      del x

      # maximum constraint dimension across time steps
      dim = 2

      def control_limits(u):
        return np.array([u_lower - u[0], u[0] - u_upper])

      return np.where(t == T, np.zeros(dim), control_limits(u))

    params = np.array([0.0, 0.1, 0.01])  # no terminal cost
    constraints_threshold = 1.0e-3

    X0 = [
        np.zeros(4),
        np.array([0.1, 0.1, 0.1, 0.1]),
        np.array([1.0, 0.0, 0.0, 0.0]),
    ]
    U0 = [np.zeros((T, 1)), 1.0 * np.ones((T, 1))]

    for U in U0:
      for x0 in X0:

        sol = optimizers.constrained_ilqr(
            functools.partial(cost, params=params), dynamics, x0, U,
            equality_constraint=equality_constraint,
            inequality_constraint=inequality_constraint,
            constraints_threshold=constraints_threshold,
            maxiter_al=10)

        # test constraints
        X = sol[0]
        U = sol[1]
        equality_constraints = sol[5]
        inequality_constraints = sol[6]

        self.assertLess(
            np.linalg.norm(equality_constraints, ord=np.inf),
            constraints_threshold)
        self.assertLess(
            np.max(inequality_constraints[inequality_constraints > 0.0]),
            constraints_threshold)
        self.assertFalse(np.any(U < u_lower - constraints_threshold))
        self.assertFalse(np.any(U > u_upper + constraints_threshold))
        self.assertLess(
            np.linalg.norm(X[-1] - goal, ord=np.inf), constraints_threshold)

  def testConstrainedCarSolve(self):
    T = 25
    dt = 0.1
    n = 3
    m = 2
    goal = np.array([1.0, 0.0, 0.0])

    def car(x, u, t):
      del t
      return np.array([u[0] * np.cos(x[2]), u[0] * np.sin(x[2]), u[1]])

    dynamics = rk4(car, dt=dt)

    cost_args = {
        'x_stage_cost': 0.1,
        'u_stage_cost': 1.0,
    }

    def cost(x, u, t, x_stage_cost, u_stage_cost):
      delta = x - goal
      stagewise_cost = 0.5 * x_stage_cost * np.dot(
          delta, delta) + 0.5 * u_stage_cost * np.dot(u, u)
      return np.where(t == T, 0.0, stagewise_cost)

    def equality_constraint(x, u, t):
      del u

      # maximum constraint dimension across time steps
      dim = 3

      def goal_constraint(x):
        err = x - goal
        return err

      return np.where(t == T, goal_constraint(x), np.zeros(dim))

    # obstacles
    obs1 = {'px': 0.5, 'py': 0.01, 'r': 0.25}

    # control limits
    u_lower = -1.0 * np.ones(m)
    u_upper = 1.0 * np.ones(m)

    def inequality_constraint(x, u, t):
      def obstacles(x):
        return np.array([
            obs1['r'] - np.sqrt((x[0] - obs1['px'])**2.0 +
                                (x[1] - obs1['py'])**2.0)
        ])

      def control_limits(u):
        return np.concatenate((u_lower - u, u - u_upper))

      return np.where(t == T, np.concatenate((np.zeros(2 * m), obstacles(x))),
                      np.concatenate((control_limits(u), obstacles(x))))

    x0 = np.zeros(n)
    U = np.zeros((T, m))

    constraints_threshold = 1.0e-3

    sol = optimizers.constrained_ilqr(
        functools.partial(cost, **cost_args), dynamics, x0, U,
        equality_constraint=equality_constraint,
        inequality_constraint=inequality_constraint,
        constraints_threshold=constraints_threshold,
        maxiter_al=10)

    # test constraints
    X = sol[0]
    U = sol[1]
    equality_constraints = sol[5]
    inequality_constraints = sol[6]

    self.assertLess(
        np.linalg.norm(equality_constraints, ord=np.inf), constraints_threshold)
    self.assertLess(
        np.max(inequality_constraints[inequality_constraints > 0.0]),
        constraints_threshold)
    self.assertFalse(np.any(U < u_lower - constraints_threshold))
    self.assertFalse(np.any(U > u_upper + constraints_threshold))
    self.assertLess(
        np.linalg.norm(X[-1] - goal, ord=np.inf), constraints_threshold)

  def testRandomShooting1(self):
    """
    test_CEM1
    Description:
        Attempts to use the Cross Entropy Method to solve the acrobot problem from "testAcrobotSolve"
    """

    T = 50
    goal = np.array([np.pi, 0.0, 0.0, 0.0])
    dynamics = euler(acrobot, dt=0.1)

    def cost(x, u, t, params):
      delta = x - goal
      terminal_cost = 0.5 * params[0] * np.dot(delta, delta)
      stagewise_cost = 0.5 * params[1] * np.dot(
          delta, delta) + 0.5 * params[2] * np.dot(u, u)
      return np.where(t == T, terminal_cost, stagewise_cost)

    x0 = np.zeros(4)
    U = np.zeros((T, 1))
    params = np.array([1000.0, 0.1, 0.01])
    zero_input_obj = 4959.476212
    self.assertLess(
        np.abs(
            optimizers.objective(
                functools.partial(cost, params=params), dynamics, U, x0) -
            zero_input_obj), 1e-6)

    optimal_obj = 51.0
    cem_hyperparams = frozendict({
        'sampling_smoothing': 0.2,
        'evolution_smoothing': 0.1,
        'elite_portion': 0.1,
        'max_iter': 100,
        'num_samples': 20_000
    })
    X_opt, U_opt, obj = optimizers.random_shooting(
        functools.partial(cost, params=params),
        dynamics,
        x0,
        U,
        np.array([-5.0]), np.array([5.0]),
        hyperparams=cem_hyperparams,
    )
    self.assertLessEqual(obj, zero_input_obj)
    self.assertLessEqual(obj, 10*optimal_obj)
    # Approximately 234

  def testCEM1(self):
    """
    test_CEM1
    Description:
      Attempts to use the Cross Entropy Method to solve the acrobot problem from "testAcrobotSolve"
    """

    T = 50
    goal = np.array([np.pi, 0.0, 0.0, 0.0])
    dynamics = euler(acrobot, dt=0.1)

    def cost(x, u, t, params):
        delta = x - goal
        terminal_cost = 0.5 * params[0] * np.dot(delta, delta)
        stagewise_cost = 0.5 * params[1] * np.dot(
            delta, delta) + 0.5 * params[2] * np.dot(u, u)
        return np.where(t == T, terminal_cost, stagewise_cost)

    x0 = np.zeros(4)
    U = np.zeros((T, 1))
    params = np.array([1000.0, 0.1, 0.01])
    zero_input_obj = 4959.476212
    self.assertLess(
        np.abs(
            optimizers.objective(
                functools.partial(cost, params=params), dynamics, U, x0) -
            zero_input_obj), 1e-6)

    optimal_obj = 51.0
    cem_hyperparams = frozendict({
        'sampling_smoothing': 0.2,
        'evolution_smoothing': 0.1,
        'elite_portion': 0.1,
        'max_iter': 100,
        'num_samples': 20_000
    })
    X_opt, U_opt, obj = optimizers.cem(
        functools.partial(cost, params=params),
        dynamics,
        x0,
        U,
        np.array([-5.0]), np.array([5.0]),
        hyperparams=cem_hyperparams,
    )
    self.assertLessEqual(obj, zero_input_obj)
    self.assertLessEqual(obj, 10*optimal_obj)
    # Objective is Around 171

if __name__ == '__main__':
  absltest.main()
