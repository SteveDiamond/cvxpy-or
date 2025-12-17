# cvxpy-or

**Operations Research-style modeling for CVXPY**

cvxpy-or brings AMPL/Pyomo-style set-based indexing to CVXPY, enabling natural modeling of transportation, scheduling, assignment, and other classic OR problems.

## Why cvxpy-or?

Writing optimization models in raw CVXPY requires thinking in terms of matrices and indices:

```python
# Raw CVXPY - indices everywhere
ship = cp.Variable((n_warehouses, n_customers), nonneg=True)
constraints = [cp.sum(ship, axis=1) <= supply]  # Which axis is which?
print(f"Ship from W1 to C2: {ship.value[0, 1]}")  # Remember the mapping
```

With cvxpy-or, you write models using meaningful names:

```python
# cvxpy-or - named indices
ship = m.add_variable(routes, nonneg=True, name='ship')
m.add_constraint('supply', sum_by(ship, 'warehouses') <= supply)
print(f"Ship Seattle->LA: {ship.get_value(('Seattle', 'LA'))}")
```

## Key Features

::::{grid} 2
:gutter: 3

:::{grid-item-card} Native CVXPY
Variables and Parameters inherit from CVXPY classes. All CVXPY operations work seamlessly.
:::

:::{grid-item-card} Named Indexing
Use meaningful names like `('Seattle', 'LA')` instead of `[0, 1]`.
:::

:::{grid-item-card} Set Operations
Union, intersection, difference, filtering, and cross products on index sets.
:::

:::{grid-item-card} Smart Aggregation
`sum_by`, `mean_by`, `min_by`, `max_by` with automatic index inference.
:::

:::{grid-item-card} Constraint Helpers
Cardinality constraints, logical implications, mutual exclusion, and more.
:::

:::{grid-item-card} pandas Integration
Load data from DataFrames, export solutions back to DataFrames.
:::
::::

## Quick Example

```python
from cvxpy_or import Model, Set, sum_by

# Create model
m = Model(name='transportation')

# Define sets
warehouses = Set(['Seattle', 'Denver', 'Chicago'], name='warehouses')
customers = Set(['NYC', 'LA', 'Houston'], name='customers')
routes = Set.cross(warehouses, customers)

# Parameters
cost = m.add_parameter(routes, name='cost', data={
    ('Seattle', 'NYC'): 2.5, ('Seattle', 'LA'): 1.0, ('Seattle', 'Houston'): 1.8,
    ('Denver', 'NYC'): 2.0, ('Denver', 'LA'): 1.5, ('Denver', 'Houston'): 1.2,
    ('Chicago', 'NYC'): 1.0, ('Chicago', 'LA'): 2.5, ('Chicago', 'Houston'): 1.5,
})
supply = m.add_parameter(warehouses, name='supply',
                         data={'Seattle': 100, 'Denver': 80, 'Chicago': 120})
demand = m.add_parameter(customers, name='demand',
                         data={'NYC': 80, 'LA': 70, 'Houston': 50})

# Variable
ship = m.add_variable(routes, nonneg=True, name='ship')

# Constraints
m.add_constraint('supply', sum_by(ship, 'warehouses') <= supply)
m.add_constraint('demand', sum_by(ship, 'customers') >= demand)

# Objective and solve
m.minimize(cost @ ship)
m.solve()

# Display results
m.print_summary()
m.print_solution(show_zero=False)
```

## Documentation

```{toctree}
:maxdepth: 2

installation
quickstart
guide/index
examples/index
api/index
```

## License

cvxpy-or is licensed under the Apache 2.0 License.
