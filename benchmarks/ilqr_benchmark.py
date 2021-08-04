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

"""Benchmarks for iLQR."""

# pylint: disable=invalid-name

import google_benchmark as benchmark
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from trajax import optimizers
from trajax.benchmarks import util


@jax.jit
def cartpole(state, action, timestep, params=(10.0, 1.0, 0.5)):
  """Classic cartpole system.

  Args:
    state: state, (4, ) array
    action: control, (1, ) array
    timestep: scalar time
    params: tuple of (MASS_CART, MASS_POLE, LENGTH_POLE)

  Returns:
    xdot: state time derivative, (4, )
  """
  del timestep  # Unused

  mc, mp, l = params
  g = 9.81

  q = state[0:2]
  qd = state[2:]
  s = jnp.sin(q[1])
  c = jnp.cos(q[1])

  H = jnp.array([[mc + mp, mp * l * c], [mp * l * c, mp * l * l]])
  C = jnp.array([[0.0, -mp * qd[1] * l * s], [0.0, 0.0]])

  G = jnp.array([[0.0], [mp * g * l * s]])
  B = jnp.array([[1.0], [0.0]])

  CqdG = jnp.dot(C, jnp.expand_dims(qd, 1)) + G
  f = jnp.concatenate(
      (qd, jnp.squeeze(-jsp.linalg.solve(H, CqdG, sym_pos=True))))

  v = jnp.squeeze(jsp.linalg.solve(H, B, sym_pos=True))
  g = jnp.concatenate((jnp.zeros(2), v))
  xdot = f + g * action

  return xdot


def cartpole_ilqr_benchmark_setup():
  """Cartpole ilqr benchmark."""

  def angle_wrap(th):
    return (th) % (2 * jnp.pi)

  def state_wrap(s):
    return jnp.array([s[0], angle_wrap(s[1]), s[2], s[3]])

  def squish(u):
    return 5 * jnp.tanh(u)

  horizon = 50
  dt = 0.1
  eq_point = jnp.array([0, jnp.pi, 0, 0])

  def cost(x, u, t):
    err = state_wrap(x - eq_point)
    stage_cost = 0.1 * jnp.dot(err, err) + 0.01 * jnp.dot(u, u)
    final_cost = 1000 * jnp.dot(err, err)
    return jnp.where(t == horizon, final_cost, stage_cost)

  def dynamics(x, u, t):
    return x + dt * cartpole(x, squish(u), t)

  x0 = jnp.array([0.0, 0.2, 0.0, -0.1])
  def bench(x0):
    _, U, _, _, _, _, _ = optimizers.ilqr(
        cost,
        dynamics,
        x0,
        jnp.zeros((horizon, 1)),
        maxiter=1000
    )
    return U

  return bench, (x0,)


# Workaround: hold refs to benchmark-registered functions, as bindings assume
# these exist during exit cleanup
benchmarks = (util.register_jit_benchmark('cartpole_ilqr_benchmark',
                                          cartpole_ilqr_benchmark_setup),)


if __name__ == '__main__':
  benchmark.main()
