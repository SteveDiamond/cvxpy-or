# Quickstart

This guide walks through building a classic transportation problem with cvxpy-or.

## The Problem

A company has warehouses in Seattle, Denver, and Chicago that need to ship products to customers in NYC, LA, and Houston. Each warehouse has limited supply, each customer has demand that must be met, and shipping costs vary by route. **Goal**: Minimize total shipping cost.

## Step 1: Import

```python
from cvxpy_or import Model, Set, Variable, Parameter, sum_by
```

## Step 2: Define Sets

Sets define the indices for your variables and parameters:

```python
warehouses = Set(['Seattle', 'Denver', 'Chicago'], name='warehouses')
customers = Set(['NYC', 'LA', 'Houston'], name='customers')
routes = Set.cross(warehouses, customers)
```

The `routes` set contains all 9 warehouse-customer pairs.

## Step 3: Define Parameters

Parameters hold your data as dictionaries mapping indices to values:

```python
cost = Parameter(routes, data={
    ('Seattle', 'NYC'): 2.5, ('Seattle', 'LA'): 1.0, ('Seattle', 'Houston'): 1.8,
    ('Denver', 'NYC'): 2.0, ('Denver', 'LA'): 1.5, ('Denver', 'Houston'): 1.2,
    ('Chicago', 'NYC'): 1.0, ('Chicago', 'LA'): 2.5, ('Chicago', 'Houston'): 1.5,
})

supply = Parameter(warehouses, data={
    'Seattle': 100, 'Denver': 80, 'Chicago': 120
})

demand = Parameter(customers, data={
    'NYC': 80, 'LA': 70, 'Houston': 50
})
```

## Step 4: Create Model and Variable

```python
m = Model(name='transportation')
ship = m.add_variable(routes, nonneg=True, name='ship')
```

## Step 5: Add Constraints

Use `sum_by` to aggregate over dimensions:

```python
# Supply constraint: total shipped from each warehouse <= supply
m.add_constraint('supply', sum_by(ship, 'warehouses') <= supply)

# Demand constraint: total received by each customer >= demand
m.add_constraint('demand', sum_by(ship, 'customers') >= demand)
```

`sum_by(ship, 'warehouses')` sums over customers for each warehouse, giving a result indexed by `warehouses`.

## Step 6: Solve and View Results

```python
m.minimize(cost @ ship)
m.solve()

m.print_summary()
m.print_solution(show_zero=False)

# Access specific values
print(f"Ship Seattle->LA: {ship.get_value(('Seattle', 'LA'))}")
```

## Complete Code

```python
from cvxpy_or import Model, Set, Variable, Parameter, sum_by

# Sets
warehouses = Set(['Seattle', 'Denver', 'Chicago'], name='warehouses')
customers = Set(['NYC', 'LA', 'Houston'], name='customers')
routes = Set.cross(warehouses, customers)

# Parameters
cost = Parameter(routes, data={
    ('Seattle', 'NYC'): 2.5, ('Seattle', 'LA'): 1.0, ('Seattle', 'Houston'): 1.8,
    ('Denver', 'NYC'): 2.0, ('Denver', 'LA'): 1.5, ('Denver', 'Houston'): 1.2,
    ('Chicago', 'NYC'): 1.0, ('Chicago', 'LA'): 2.5, ('Chicago', 'Houston'): 1.5,
})
supply = Parameter(warehouses, data={
    'Seattle': 100, 'Denver': 80, 'Chicago': 120
})
demand = Parameter(customers, data={
    'NYC': 80, 'LA': 70, 'Houston': 50
})

# Model
m = Model(name='transportation')
ship = m.add_variable(routes, nonneg=True, name='ship')

# Constraints
m.add_constraint('supply', sum_by(ship, 'warehouses') <= supply)
m.add_constraint('demand', sum_by(ship, 'customers') >= demand)

# Solve
m.minimize(cost @ ship)
m.solve()
m.print_summary()
m.print_solution(show_zero=False)
```

## What's Next?

- Learn about [Sets, Variables, and Parameters](guide/basic-usage.md)
- Explore [aggregation functions](guide/aggregations.md) like `mean_by`, `min_by`, `max_by`
- See [constraint helpers](guide/constraints.md) for cardinality and logical constraints
- Load data from files with [pandas I/O](guide/pandas-io.md) or [xarray I/O](guide/xarray-io.md)
- Browse [more examples](examples/index.md) including assignment, facility location, and more
