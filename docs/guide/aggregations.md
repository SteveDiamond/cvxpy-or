# Aggregations

cvxpy-or provides aggregation functions that work with named indices, making it easy to sum, average, or find extremes over specific dimensions.

## sum_by

`sum_by` aggregates an expression by grouping on specified index positions.

### Basic Usage

```python
from cvxpy_or import Set, Variable, sum_by

# Create a route-indexed variable
warehouses = Set(['Seattle', 'Denver', 'Chicago'], name='warehouses')
customers = Set(['NYC', 'LA', 'Houston'], name='customers')
routes = Set.cross(warehouses, customers)

ship = Variable(routes, nonneg=True, name='ship')

# Sum over customers for each warehouse
total_shipped = sum_by(ship, 'warehouses')
# Result is indexed by warehouses: {Seattle: ..., Denver: ..., Chicago: ...}

# Sum over warehouses for each customer
total_received = sum_by(ship, 'customers')
# Result is indexed by customers: {NYC: ..., LA: ..., Houston: ...}
```

### Understanding the Grouping

When you call `sum_by(expr, 'dim')`, you're saying "keep this dimension, sum over the others":

```python
# routes has dimensions: (warehouses, customers)

sum_by(ship, 'warehouses')
# Groups by warehouse, sums over customers
# Seattle gets: ship[Seattle,NYC] + ship[Seattle,LA] + ship[Seattle,Houston]

sum_by(ship, 'customers')
# Groups by customer, sums over warehouses
# NYC gets: ship[Seattle,NYC] + ship[Denver,NYC] + ship[Chicago,NYC]
```

### Keeping Multiple Dimensions

```python
# Three-dimensional index
periods = Set(['Jan', 'Feb'], name='periods')
routes_periods = Set.cross(warehouses, customers, periods)
ship = Variable(routes_periods, nonneg=True)

# Sum over periods, keep warehouse and customer
sum_by(ship, ['warehouses', 'customers'])

# Sum over customers and periods, keep warehouse only
sum_by(ship, 'warehouses')
```

### Total Sum

To get a single total, don't specify any positions:

```python
total = sum_by(ship)  # Sum over all dimensions
```

## mean_by

`mean_by` computes the mean by grouping on specified positions.

```python
from cvxpy_or import mean_by

# Average cost per warehouse
avg_cost = mean_by(cost, 'warehouses')

# Average across all
overall_avg = mean_by(cost)
```

## min_by and max_by

These functions find the minimum or maximum over groups. Unlike `sum_by` and `mean_by`, they return both a variable and constraints (because min/max require auxiliary variables in convex optimization).

```python
from cvxpy_or import min_by, max_by

# Maximum shipment from each warehouse
max_ship, constraints = max_by(ship, 'warehouses')
m.add_constraint('max_constraints', constraints)

# Minimum shipment to each customer
min_ship, constraints = min_by(ship, 'customers')
m.add_constraint('min_constraints', constraints)
```

### Using min_by/max_by in Objectives

```python
# Minimize the maximum load (load balancing)
max_load, constraints = max_by(load, 'servers')
m.add_constraint('max_load_def', constraints)
m.minimize(max_load)
```

## count_by

Count the number of elements in each group (useful for understanding index structure):

```python
from cvxpy_or import count_by

# How many customers does each warehouse serve?
counts = count_by(routes, 'warehouses')
# {'Seattle': 3, 'Denver': 3, 'Chicago': 3}
```

## group_keys

Get the unique keys for a grouping:

```python
from cvxpy_or import group_keys

# Get unique warehouses from routes
keys = group_keys(routes, 'warehouses')
# ['Seattle', 'Denver', 'Chicago']
```

## Index Inference

Aggregation functions automatically infer the index from expressions:

```python
# Works with expressions, not just variables
total_cost = sum_by(cost * ship, 'warehouses')

# The index is inferred from the expression
```

## Common Patterns

### Supply/Demand Constraints

```python
# Supply: sum shipped from each warehouse <= supply
m.add_constraint('supply', sum_by(ship, 'warehouses') <= supply)

# Demand: sum received by each customer >= demand
m.add_constraint('demand', sum_by(ship, 'customers') >= demand)
```

### Cost Calculations

```python
# Total cost per warehouse
warehouse_cost = sum_by(cost * ship, 'warehouses')

# Total cost per customer
customer_cost = sum_by(cost * ship, 'customers')

# Grand total
total_cost = sum_by(cost * ship)
```

### Balancing Constraints

```python
# Flow balance at each node
# inflow = outflow
m.add_constraint('balance',
    sum_by(inflow, 'nodes') == sum_by(outflow, 'nodes'))
```

### Capacity Constraints

```python
# Total assigned to each resource <= capacity
m.add_constraint('capacity',
    sum_by(assignment, 'resources') <= capacity)
```
