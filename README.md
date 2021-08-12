# trajax

A Python library for differentiable optimal control on accelerators.

Jump to: [**installation**](#installation)
| [**background**](#trajectory-optimization-and-optimal-control)
| [**API**](#api)
| [**limitations**](#limitations)

Trajax builds on [JAX](https://github.com/google/jax) and hence code written
with Trajax supports JAX's transformations. In particular, Trajax's solvers:

1. Are automatically efficiently differentiable, via `jax.grad`.
2. Scale up to parallel instances via `jax.vmap` and `jax.pmap`.
3. Can run on CPUs, GPUs, and TPUs without code changes, and support end-to-end compilation with `jax.jit`.
4. Are made available from Python, written with NumPy.

In Trajax, differentiation through the solution of a trajectory optimization problem is done more efficiently than by differentiating the solver implementation directly. Specifically, Trajax defines custom differentiation routines for its solvers. It registers these with JAX so that they are picked up whenever using JAX's autodiff features (e.g. `jax.grad`) to differentiate functions that call a Trajax solver.

**This is a research project, not an official Google product.**

Trajax is currently a work in progress, maintained by a few individuals at Google Research. While we are actively using Trajax in our own research projects, expect there to be bugs and rough edges compared to commercially available solvers.

## Installation

To install directly from github using `pip`:

```bash
$ pip install git+https://github.com/google/trajax
```

Alternatively, to install from source:

```bash
$ python setup.py install
```

## Trajectory optimization and optimal control

We consider classical optimal control tasks concerning optimizing trajectories of a given discrete time dynamical system by solving the following problem. Given a cost function `c`, dynamics function `f`, and initial state `x0`, the goal is to compute:

```python
argmin(lambda X, U: sum(c(X[t], U[t], t) for t in range(T)) + c_final(X[T]))
```

subject to the constraint that `X[0] == x0` and that:

```python
all(X[t + 1] == f(X[t], U[t], t) for t in range(T))
```

There are many resources for more on trajectory optimization, including [_Dynamic Programming and Optimal Control_ by Dimitri Bertsekas](http://athenasc.com/dpbook.html) and [_Underactuated Robotics_ by Russ Tedrake](http://underactuated.mit.edu/trajopt.html).

## API

In describing the API, it will be useful to abbreviate a JAX/NumPy floating point ndarray of shape `(a, b, …)` as a type denoted `F[a, b, …]`. Assume `n` is the state dimension, `d` is the control dimension, and `T` is the time horizon.

### Problem setup convention/signature

Setting up a problem requires writing two functions, cost and dynamics, with type signatures:

```python
cost(state: F[n], action: F[d], time_step: int) : float
dynamics(state: F[n], action: F[d], time_step: int) : F[n]
```

Note that even if a dimension `n` or `d` is 1, the corresponding state or action representation is still a rank-1 ndarray (i.e. a vector, of length 1).

Because Trajax uses JAX, the `cost` and `dynamics` functions must be written in a functional programming style as required by JAX. See [the JAX readme](https://github.com/google/jax#current-gotchas) for details on writing JAX-friendly functional code. By and large, functions that have no side effects and that use `jax.numpy` in place of `numpy` are likely to work.

### Solvers

If we abbreviate the type of the above two functions as `CostFn` and `DynamicsFn`, then our solvers have the following type signature prefix in common:

```python
solver(cost: CostFn, dynamics: DynamicsFn, initial_state: F[n], initial_actions: F[T, d], *solver_args, **solver_kwargs): SolverOutput
```

`SolverOutput` is a tuple of `(F[T + 1, n], F[T, d], float, *solver_outputs)`. The first three tuple components represent the optimal state trajectory, optimal control sequence, and the optimal objective value achieved, respectively. The remaining `*solver_outputs` are specific to the particular solver (such as number of iterations, norm of the final gradient, etc.).

There are currently four solvers provided: `ilqr`, `scipy_minimize`, `cem`, and `random_shooting`. Each extends the signatures above with solver-specific arguments and output values. Details are provided in each solver function's docstring.

Underlying the `ilqr` implementation is a time-varying LQR routine, which solves a special case of the above problem, where costs are convex quadratic and dynamics are affine. To capture this, both are represented as matrices. This routine is also made available as `tvlqr`.

### Objectives

One might want to write a custom solver, or work with an objective function for any other reason. To that end, Trajax offers the optimal control objective in the form of an API function:

```python
objective(cost: CostFn, dynamics: DynamicsFn, initial_state: F[n], actions: F[T, d]): float
```

Combining this function with JAX's autodiff capabilities offers, for example, a starting point for writing a first-order custom solver. For example:

```python
def improve_controls(cost, dynamics, U, x0, eta, num_iters):
  grad_fn = jax.grad(trajax.objective, argnums=(2,))
  for i in range(num_iters):
    U = U - eta * grad_fn(cost, dynamics, U, x0)
  return U
```

The solvers provided by Trajax are actually built around this `objective` function. For instance, the `scipy_minimize` solver simply calls `scipy.minimize.minimize` with the gradient and Hessian-vector product functions derived from `objective` using `jax.grad` and `jax.hessian`.

## Limitations

​​Just as Trajax inherits the autodiff, compilation, and parallelism features of JAX, it also inherits its corresponding limitations. Functions such as the cost and dynamics given to a solver must be written using `jax.numpy` in place of standard `numpy`, and must conform to a functional style; see [the JAX readme](https://github.com/google/jax#current-gotchas). Due to the complexity of trajectory optimizer implementations, initial compilation times can be long.
