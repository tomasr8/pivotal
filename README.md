<p align="center">
    <img src="https://raw.githubusercontent.com/tomasr8/pivotal/master/logo.svg">
</p>

# No fuss Linear & Integer Programming solver

```python
from pivotal import minimize, maximize, Variable

x = Variable("x")
y = Variable("y")
z = Variable("z")

objective = 2*x + y + 3*z
constraints = (
    x - y == 4,
    y + 2*z == 2
)

minimize(objective, constraints)
# -> value: 11.0
# -> variables: {'x': 4.0, 'y': 0.0, 'z': 1.0}

maximize(objective, constraints)
# -> value: 14.0
# -> variables: {'x': 6.0, 'y': 2.0, 'z': 0.0}
```

## About

`Pivotal` is not aiming to compete with commerical solvers like Gurobi. Rather, it is aiming to simplify the process of creating and solving linear programs thanks to its very simple and intuitive API. The LP solver uses a 2-phase Simplex algorithm, and MILP problems are solved using branch & bound with LP relaxations.

## Installation

Python >=3.11 is required.

Install via pip:

```bash
pip install pivotal-solver
```

## API

### Variables

`Variable` instances implement `__add__`, `__sub__` and other magic methods, so you can use them directly in expressions such as `2*x + 10 - y`.

Here are some examples of what you can do with them:

```python
x = Variable("x")
y = Variable("y")
z = Variable("z")

2*x + 10 - y
x + (y - z)*10
-x
-(x + y)
sum([x, y, z])
abs(x)
abs(x - 2)

X = [Variable(f"x{i}") for i in range(5)]
sum(X)
```

Note that variables are considered equal if they have the same name, so
for example this expression:

```python
Variable("x") + 2 + Variable("x")
```

will be treated as simply `2*x+2`.

The first argument to `minimize` and `maximize` is the objective function which must be either a single variable or a linear combination as in the examples above.

#### Variable bounds

By default, variables are assumed to be nonnegative, but arbitrary lower and upper bounds are supported:

```python
# Default: x >= 0, same as (lower=0, upper=None)
x = Variable("x")

# Lower bound: x >= 5
y = Variable("y", lower=5)

# Upper bound: 0 <= z <= 10
z = Variable("z", upper=10)

# Both bounds: -5 <= w <= 5
w = Variable("w", lower=-5, upper=5)

# Free variable (unbounded): -∞ < v < ∞
v = Variable("v", lower=None, upper=None)
```

Example:

```python
from pivotal import minimize, Variable

# Minimize 2*x + y subject to x + y >= 10
# with bounds: 3 <= x <= 7, y >= 0
x = Variable("x", lower=3, upper=7)
y = Variable("y")

result = minimize(2*x + y, (x + y >= 10,))
# -> value: 13.0
# -> variables: {'x': 3.0, 'y': 7.0}
```

#### Integer & binary variables

Variables can be declared as integer or binary for mixed-integer linear programming (MILP):

```python
# Integer variable: must take integer values
n = Variable("n", var_type="integer")

# Integer variable with bounds: 0 <= k <= 10, integer
k = Variable("k", var_type="integer", upper=10)

# Binary variable: 0 or 1 (bounds are enforced automatically)
b = Variable("b", var_type="binary")
```

When any variable in the problem is integer or binary, the solver automatically switches to branch & bound. The existing `minimize`/`maximize` API works unchanged:

```python
from pivotal import minimize, Variable

x = Variable("x", var_type="integer")
y = Variable("y")  # continuous

minimize(x + y, (x + y >= 3.5, y <= 1))
# -> value: 3.5
# -> variables: {'x': 3.0, 'y': 0.5}
```

A classic 0-1 knapsack:

```python
from pivotal import maximize, Variable

items = [Variable(f"x{i}", var_type="binary") for i in range(4)]
weights = [2, 3, 4, 5]
values = [3, 4, 5, 6]

maximize(
    sum(values[i] * items[i] for i in range(4)),
    (sum(weights[i] * items[i] for i in range(4)) <= 8,),
)
# -> value: 10.0
# -> variables: {'x0': 0.0, 'x1': 1.0, 'x2': 0.0, 'x3': 1.0}
```

### Constraints

There are three supported constraints: `==` (equality), `>=` (greater than or equal) and `<=` (less than or equal). You create a constraint simply by using these comparisons in expressions involving `Variable` instances. For example:

```python
x = Variable("x")
y = Variable("y")
z = Variable("z")

x == 4
2*x - y == z + 7
y >= -x + 3*z
x <= 0
```

There is no need to convert your constraints to the canonical form which uses only equality constraints. This is done automatically by the solver.

`minimize` and `maximize` expect a list of constraints as the second argument.

### Output

The return value of `minimize` and `maximize` is a 2-tuple containing the value of the objective function and a dictionary of variables and their values.

The functions may raise `pivotal.Infeasible` if the program is over-constrained (no solution exists) or `pivotal.Unbounded` if the program is under-constrained (the objective can be made arbitrarily small):

```python
from pivotal import minimize, maximize, Variable, Infeasible

x = Variable("x")
y = Variable("y")

objective = 2*x + y
constraints = (
    x + 2*y == 4,
    x + y == 10
)

try:
    minimize(objective, constraints)
except Infeasible:
    print("No solution")
```

### Absolute values

Absolute values are supported in both the objective and constraints.
You can use:

- `min |expr|` and `max |-expr|` in the objective (`min -|expr|` and `max |expr|` cannot be solved with pure LP solvers and require MILP solvers instead, see [link](https://optimization.cbe.cornell.edu/index.php?title=Optimization_with_absolute_values))
- `|expr| ≤ C` and `|expr| = 0` in the constraints (`|expr| ≥ C` and `|expr| = C` cannot be solved with pure LP solvers and require MILP solvers instead, see [link](https://optimization.cbe.cornell.edu/index.php?title=Optimization_with_absolute_values))

```python
from pivotal import minimize, Variable

x = Variable("x")

objective = abs(x - 5)
constraints = (
    abs(x) <= 5,
)

minimize(objective, constraints)
# -> value: 0.0
# -> variables: {'x': 5.0}
```

### Iterations, Tolerance & Node Limit

`minimize` and `maximize` take the following keyword arguments:

- `max_iterations` (default `math.inf`): maximum number of iterations of the second phase of the Simplex algorithm. If the maximum number of iterations is reached, a potentially non-optimal solution is returned.
- `tolerance` (default `1e-6`): precision of floating point comparisons, e.g. when comparing against zero. Instead of `x == 0.0`, the algorithm considers a value to be zero when it is within the given tolerance: `abs(x) <= tolerance`. Also used for integrality checks in MILP.
- `node_limit` (default `10_000`): maximum number of branch & bound nodes to explore when solving MILPs. If the limit is reached and a feasible solution has been found, the best solution so far is returned. Otherwise, `pivotal.NodeLimitReached` is raised.

## Development

### Setting up

```bash
git clone https://github.com/tomasr8/pivotal.git
cd pivotal
uv sync --group dev
```

### Running tests

```bash
pytest
```
