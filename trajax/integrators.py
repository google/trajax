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

"""Simple helpers for converting continuous-time dynamics to discrete-time."""


def euler(dynamics, dt=0.01):
  return lambda x, u, t: x + dt * dynamics(x, u, t)


def rk4(dynamics, dt=0.01):
  def integrator(x, u, t):
    dt2 = dt / 2.0
    k1 = dynamics(x, u, t)
    k2 = dynamics(x + dt2 * k1, u, t)
    k3 = dynamics(x + dt2 * k2, u, t)
    k4 = dynamics(x + dt * k3, u, t)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
  return integrator
