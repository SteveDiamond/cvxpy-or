# Quickstart

This guide walks through building a classic transportation problem with cvxpy-or.

## The Problem

A company has warehouses in Seattle, Denver, and Chicago that need to ship products to customers in NYC, LA, and Houston. Each warehouse has limited supply, each customer has demand that must be met, and shipping costs vary by route. **Goal**: Minimize total shipping cost.

## Step 1: Create the Model

```python
from cvxpy_or import Model, Set, sum_by

m = Model(name='transportation')
```

The `Model` class provides a clean interface for building optimization problems.

## Step 2: Define Sets

Sets define the indices for your variables and parameters:

```python
# Simple sets
warehouses = Set(['Seattle', 'Denver', 'Chicago'], name='warehouses')
customers = Set(['NYC', 'LA', 'Houston'], name='customers')

# Compound set (cross product)
routes = Set.cross(warehouses, customers)
```

The `routes` set contains all 9 warehouse-customer pairs:
```
('Seattle', 'NYC'), ('Seattle', 'LA'), ('Seattle', 'Houston'),
('Denver', 'NYC'), ('Denver', 'LA'), ('Denver', 'Houston'),
('Chicago', 'NYC'), ('Chicago', 'LA'), ('Chicago', 'Houston')
```

## Step 3: Add Parameters

Parameters hold your input data:

```python
# Shipping cost per unit for each route
cost = m.add_parameter(routes, name='cost', data={
    ('Seattle', 'NYC'): 2.5, ('Seattle', 'LA'): 1.0, ('Seattle', 'Houston'): 1.8,
    ('Denver', 'NYC'): 2.0, ('Denver', 'LA'): 1.5, ('Denver', 'Houston'): 1.2,
    ('Chicago', 'NYC'): 1.0, ('Chicago', 'LA'): 2.5, ('Chicago', 'Houston'): 1.5,
})

# Supply at each warehouse
supply = m.add_parameter(warehouses, name='supply',
                         data={'Seattle': 100, 'Denver': 80, 'Chicago': 120})

# Demand at each customer
demand = m.add_parameter(customers, name='demand',
                         data={'NYC': 80, 'LA': 70, 'Houston': 50})
```

## Step 4: Add Variables

Variables are what the solver determines:

```python
# Amount to ship on each route (non-negative)
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

## Step 6: Set Objective

```python
# Minimize total shipping cost
m.minimize(cost @ ship)
```

The `@` operator computes the dot product (element-wise multiply then sum).

## Step 7: Solve

```python
m.solve()
```

## Step 8: View Results

```python
# Summary
m.print_summary()

# Solution details
m.print_solution(show_zero=False)

# Access specific values
print(f"Ship Seattle->LA: {ship.get_value(('Seattle', 'LA'))}")
```

## Complete Code

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

# Objective
m.minimize(cost @ ship)

# Solve and display
m.solve()
m.print_summary()
m.print_solution(show_zero=False)
```

## What's Next?

- Learn about [Sets, Variables, and Parameters](guide/basic-usage.md)
- Explore [aggregation functions](guide/aggregations.md) like `mean_by`, `min_by`, `max_by`
- See [constraint helpers](guide/constraints.md) for cardinality and logical constraints
- Browse [more examples](examples/index.md) including assignment, facility location, and more
