# cvxpy-or

Operations Research-style modeling for CVXPY.

This package provides AMPL/Pyomo-style set-based indexing for CVXPY, enabling natural modeling of transportation, scheduling, and other OR problems.

## Installation

```bash
pip install cvxpy-or

# Optional dependencies
pip install cvxpy-or[pandas]   # For DataFrame I/O
pip install cvxpy-or[display]  # For rich table output
pip install cvxpy-or[all]      # Everything
```

With uv:
```bash
uv add cvxpy-or
```

## Quick Start

```python
from cvxpy_or import Model, Set, sum_by

# Create model
m = Model(name='transportation')

# Define sets
m.warehouses = Set(['Seattle', 'Denver', 'Chicago'], name='warehouses')
m.customers = Set(['NYC', 'LA', 'Houston'], name='customers')
m.routes = Set.cross(m.warehouses, m.customers)

# Parameters
m.cost = m.add_parameter(m.routes, name='cost', data={
    ('Seattle', 'NYC'): 2.5, ('Seattle', 'LA'): 1.0, ('Seattle', 'Houston'): 1.8,
    ('Denver', 'NYC'): 2.0, ('Denver', 'LA'): 1.5, ('Denver', 'Houston'): 1.2,
    ('Chicago', 'NYC'): 1.0, ('Chicago', 'LA'): 2.5, ('Chicago', 'Houston'): 1.5,
})
m.supply = m.add_parameter(m.warehouses, name='supply',
                           data={'Seattle': 100, 'Denver': 80, 'Chicago': 120})
m.demand = m.add_parameter(m.customers, name='demand',
                           data={'NYC': 80, 'LA': 70, 'Houston': 50})

# Variable
m.ship = m.add_variable(m.routes, nonneg=True, name='ship')

# Constraints
m.add_constraint('supply', sum_by(m.ship, 'warehouses') <= m.supply)
m.add_constraint('demand', sum_by(m.ship, 'customers') >= m.demand)

# Objective
m.minimize(m.cost @ m.ship)

# Solve and display
m.solve()
m.print_summary()
m.print_solution(show_zero=False)
```

## Key Features

- **Native CVXPY**: `Variable` and `Parameter` inherit from CVXPY - all operations work
- **Model wrapper**: Clean interface for building problems
- **Set operations**: Union, intersection, difference, filtering
- **Named aggregation**: `sum_by`, `mean_by`, `min_by`, `max_by`
- **Constraint helpers**: `at_most_k`, `exactly_k`, `implies`, `mutex`, `one_of`
- **Validation**: Helpful error messages with typo suggestions
- **pandas I/O**: Load data from DataFrames, export solutions
- **Pretty printing**: Rich tables for variables and solutions

## API Reference

### Core Classes

#### `Set(elements, name=None, names=None)`

An ordered set of elements for indexing.

```python
warehouses = Set(['W1', 'W2', 'W3'], name='warehouses')
customers = Set(['C1', 'C2', 'C3'], name='customers')

# Cross product
routes = Set.cross(warehouses, customers)

# Set operations
A | B  # Union
A & B  # Intersection
A - B  # Difference
A ^ B  # Symmetric difference
A <= B # Subset

# Filtering and transformation
evens = numbers.filter(lambda x: x % 2 == 0)
doubled = numbers.map(lambda x: x * 2)
sorted_set = numbers.sorted()

# Access
s.first(), s.last()
len(s), 'W1' in s
```

#### `Variable(index, nonneg=False, name=None, **kwargs)`

A CVXPY Variable indexed by a Set.

```python
ship = Variable(routes, nonneg=True, name='ship')
ship[('W1', 'C1')]          # Access by key
ship.get_value(('W1', 'C1')) # Get solved value
```

#### `Parameter(index, data=None, name=None, **kwargs)`

A CVXPY Parameter indexed by a Set.

```python
cost = Parameter(routes, data={('W1', 'C1'): 10, ...})
cost.set_data(new_data)     # Update values
cost.expand(larger_index, positions)  # Broadcast
```

#### `Model(name=None)`

Wrapper for building optimization problems.

```python
m = Model(name='my_problem')

# Create components
x = m.add_variable(index, name='x', nonneg=True)
p = m.add_parameter(index, data={...}, name='p')

# Add constraints
m.add_constraint('bounds', x <= 100)

# Set objective
m.minimize(cost @ x)  # or m.maximize(...)

# Solve
status = m.solve()
m.print_summary()
m.print_solution()

# Access results
m.status, m.value
m.get_variable('x'), m.get_parameter('p')
df = m.to_dataframe()
```

### Aggregation Functions

```python
from cvxpy_or import sum_by, mean_by, min_by, max_by, count_by, group_keys

# Sum by position (most common)
sum_by(ship, 'warehouses')  # Sum over customers for each warehouse
sum_by(ship, ['origin', 'period'])  # Keep multiple dimensions

# Mean by position
mean_by(cost, 'warehouses')  # Average cost per warehouse

# Min/max by position (returns variable + constraints)
max_ship, constraints = max_by(ship, 'warehouses')
min_ship, constraints = min_by(ship, 'customers')

# Utilities
counts = count_by(routes, 'warehouses')  # Elements per group
keys = group_keys(routes, 'warehouses')  # Unique group keys
```

### Filtering

```python
from cvxpy_or import where

# Filter expression elements
where(ship, lambda r: r[0] == 'W1')  # Callable
where(ship, origin='W1')             # Keyword
where(ship, origin=['W1', 'W2'])     # Multiple values
```

### Constraint Helpers

```python
from cvxpy_or import at_most_k, at_least_k, exactly_k, implies, mutex, one_of, bounds

# Cardinality constraints (returns list of constraints)
constraints = at_most_k(x, k=3)   # At most 3 nonzero
constraints = exactly_k(x, k=3)  # Exactly 3 nonzero
constraints = at_least_k(x, k=2) # At least 2 nonzero

# Logical constraints for binary variables
implies(a, b)  # a=1 => b=1
mutex(a, b, c) # At most one is 1
one_of(a, b, c) # Exactly one is 1

# Bounds from parameters
constraints = bounds(ship, lower=0, upper=capacity)
```

### Validation

```python
from cvxpy_or import validate_keys, validate_numeric, validate_bounds, ValidationError

# Validate data keys match index (with helpful error messages)
validate_keys(data, index)  # Raises ValidationError with suggestions

# Validate data types and bounds
validate_numeric(data)
validate_bounds(data, lower=0, upper=100)
```

### pandas I/O

```python
from cvxpy_or import (
    set_from_series, set_from_dataframe, set_from_index,
    parameter_from_dataframe, parameter_from_series,
    variable_to_dataframe, parameter_to_dataframe
)

# Create Set from DataFrame
customers = set_from_series(df['customer_id'])
routes = set_from_dataframe(df, columns=['origin', 'dest'])

# Create Parameter from DataFrame
cost = parameter_from_dataframe(df, index_cols=['origin', 'dest'], value_col='cost')
supply = parameter_from_series(df.set_index('warehouse')['supply'])

# Export solutions to DataFrame
df = variable_to_dataframe(ship)
df = m.to_dataframe('ship')  # From Model
```

### Display

```python
from cvxpy_or import print_variable, print_solution, variable_table

# Print variable values (after solving)
print_variable(ship, show_zero=False)

# Print solution summary
print_solution([ship, inventory], objective_value=m.value, status=m.status)

# Get table as string
table_str = variable_table(ship, precision=2)
```

## Examples

See the `examples/` directory for complete examples:

- `assignment_problem.py` - Worker-task assignment
- `blending_problem.py` - Blend optimization
- `diet_problem.py` - Classic diet problem
- `facility_location.py` - Facility location (UFLP)
- `multi_period_transportation.py` - Multi-period with inventory

## Comparison: cvxpy-or vs Raw CVXPY

**Raw CVXPY:**
```python
import cvxpy as cp
import numpy as np

n_warehouses, n_customers = 3, 4
cost = np.array([[1, 2, 3, 4], [2, 1, 2, 3], [3, 2, 1, 2]])
supply = np.array([100, 80, 120])
demand = np.array([60, 70, 50, 40])

ship = cp.Variable((n_warehouses, n_customers), nonneg=True)
prob = cp.Problem(
    cp.Minimize(cp.sum(cp.multiply(cost, ship))),
    [
        cp.sum(ship, axis=1) <= supply,
        cp.sum(ship, axis=0) >= demand,
    ]
)
prob.solve()

# Accessing results requires remembering indices
print(f"Ship from W1 to C2: {ship.value[0, 1]}")
```

**cvxpy-or:**
```python
from cvxpy_or import Model, Set, sum_by

m = Model()
warehouses = Set(['Seattle', 'Denver', 'Chicago'], name='warehouses')
customers = Set(['NYC', 'LA', 'Houston', 'Miami'], name='customers')
routes = Set.cross(warehouses, customers)

cost = m.add_parameter(routes, data={('Seattle', 'NYC'): 1, ...})
supply = m.add_parameter(warehouses, data={'Seattle': 100, ...})
demand = m.add_parameter(customers, data={'NYC': 60, ...})
ship = m.add_variable(routes, nonneg=True, name='ship')

m.add_constraint('supply', sum_by(ship, 'warehouses') <= supply)
m.add_constraint('demand', sum_by(ship, 'customers') >= demand)
m.minimize(cost @ ship)
m.solve()

# Named access to results
print(f"Ship from Seattle to LA: {ship.get_value(('Seattle', 'LA'))}")
m.print_solution()
```

## License

Apache-2.0
