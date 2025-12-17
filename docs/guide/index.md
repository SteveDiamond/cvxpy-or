# User Guide

These guides cover the core concepts and features of cvxpy-or.

```{toctree}
:maxdepth: 2

basic-usage
aggregations
constraints
pandas-io
validation
```

## Overview

cvxpy-or is built around a few key concepts:

| Concept | Description |
|---------|-------------|
| **Set** | An ordered collection of elements that serves as an index |
| **Variable** | A CVXPY Variable indexed by a Set |
| **Parameter** | A CVXPY Parameter indexed by a Set |
| **Model** | A container that manages variables, parameters, constraints, and objectives |

## Typical Workflow

1. **Create a Model** - Container for your optimization problem
2. **Define Sets** - The indices for your data and decisions
3. **Add Parameters** - Load your input data
4. **Add Variables** - Define decision variables
5. **Add Constraints** - Specify the rules
6. **Set Objective** - Define what to optimize
7. **Solve** - Find the optimal solution
8. **Inspect Results** - View and export the solution

```python
from cvxpy_or import Model, Set, sum_by

# 1. Create model
m = Model(name='my_problem')

# 2. Define sets
items = Set(['A', 'B', 'C'], name='items')

# 3. Add parameters
cost = m.add_parameter(items, data={'A': 10, 'B': 20, 'C': 15})

# 4. Add variables
x = m.add_variable(items, nonneg=True, name='x')

# 5. Add constraints
m.add_constraint('budget', cost @ x <= 100)

# 6. Set objective
m.maximize(sum_by(x))

# 7. Solve
m.solve()

# 8. Inspect results
m.print_solution()
```
