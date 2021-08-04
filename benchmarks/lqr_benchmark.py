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
"""Benchmarks for LQR-related procedures."""

# pylint: disable=invalid-name


import google_benchmark as benchmark
from trajax import tvlqr
from trajax.benchmarks import util
import numpy as onp


# Workaround: hold refs to benchmark-registered functions, as bindings assume
# these exist during exit cleanup
_bench_fns = []


def register_lqr_rollout_jit_benchmark(name, state_dim, control_dim,
                                       time_horizon):
  """Generate and register an LQR rollout benchmark with the given shapes."""
  n, d, T = state_dim, control_dim, time_horizon

  def lqr_rollout_setup():
    def bench(K, k, x0, A, B, c):
      _, U = tvlqr.rollout(K, k, x0, A, B, c)
      return U

    onp_rng = onp.random.RandomState(0)
    A = onp_rng.randn(T, n, n)
    B = onp_rng.randn(T, n, d)
    c = onp_rng.randn(T, n)
    x0 = onp_rng.randn(n)
    K = onp_rng.randn(T, d, n)
    k = onp_rng.randn(T, d)

    return bench, (K, k, x0, A, B, c)

  bench = util.register_jit_benchmark(name, lqr_rollout_setup)
  _bench_fns.append(bench)


register_lqr_rollout_jit_benchmark('lqr_rollout_2x2x50', 2, 2, 50)
register_lqr_rollout_jit_benchmark('lqr_rollout_2x5x50', 2, 5, 50)
register_lqr_rollout_jit_benchmark('lqr_rollout_20x30x100', 20, 30, 100)


if __name__ == '__main__':
  benchmark.main()
