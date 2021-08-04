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

"""Utilities for writing benchmarks."""

import google_benchmark as benchmark
import jax


def register_jit_benchmark(name, setup):
  """Register a benchmark under a top-level jit, with arguments on device."""
  def bench(state):
    f, args = setup()
    f = jax.jit(f)
    args = [jax.device_put(x) for x in args]

    # call once for a jit warm-up, and to detect multiple outputs
    outs = f(*args)
    multiple_outputs = isinstance(outs, tuple)
    del outs

    if multiple_outputs:
      while state:
        outs = f(*args)
        for out in outs:
          out.block_until_ready()
    else:
      while state:
        f(*args).block_until_ready()

  bench.__name__ = name
  return benchmark.register(bench)
