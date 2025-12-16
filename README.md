# cvxpy-or

Operations Research-style modeling for CVXPY.

This package provides AMPL/Pyomo-style set-based indexing for CVXPY, enabling natural modeling of transportation, scheduling, and other OR problems.

## Installation

```bash
pip install cvxpy-or
```

## Quick Start

```python
from cvxpy_or import Set, Variable, Parameter, sum_by
import cvxpy as cp

# Define index sets
warehouses = Set(['Seattle', 'Denver', 'Chicago'], name='warehouses')
customers = Set(['NYC', 'LA', 'Houston'], name='customers')
routes = Set.cross(warehouses, customers, name='routes')

# Create indexed variables and parameters
cost = Parameter(routes, data={
    ('Seattle', 'NYC'): 2.5, ('Seattle', 'LA'): 1.0, ('Seattle', 'Houston'): 1.8,
    ('Denver', 'NYC'): 2.0, ('Denver', 'LA'): 1.5, ('Denver', 'Houston'): 1.2,
    ('Chicago', 'NYC'): 1.0, ('Chicago', 'LA'): 2.5, ('Chicago', 'Houston'): 1.5,
})
supply = Parameter(warehouses, data={'Seattle': 100, 'Denver': 80, 'Chicago': 120})
demand = Parameter(customers, data={'NYC': 80, 'LA': 70, 'Houston': 50})

ship = Variable(routes, nonneg=True)

# Build problem using native CVXPY operations
prob = cp.Problem(
    cp.Minimize(cost @ ship),
    [
        sum_by(ship, 'warehouses') <= supply,
        sum_by(ship, 'customers') >= demand,
    ]
)
prob.solve()

print(f"Optimal cost: ${prob.value:.2f}")
```

## Key Features

- **Native CVXPY**: `Variable` and `Parameter` inherit from CVXPY classes - all operations work natively
- **Set-based indexing**: Define index sets and cross-products naturally
- **Named aggregation**: `sum_by('warehouses')` instead of manual matrix operations
- **No for loops**: Express constraints over entire index sets at once

## API

### `Set(elements, name=None, names=None)`

An ordered set of elements for indexing.

```python
warehouses = Set(['W1', 'W2', 'W3'], name='warehouses')
routes = Set.cross(warehouses, customers, name='routes')  # Cross product
```

### `Variable(index, nonneg=False, name=None, **kwargs)`

A CVXPY Variable indexed by a Set.

```python
ship = Variable(routes, nonneg=True, name='ship')
ship[('W1', 'C1')]  # Access by index key
```

### `Parameter(index, data=None, name=None, **kwargs)`

A CVXPY Parameter indexed by a Set.

```python
cost = Parameter(routes, data={('W1', 'C1'): 10, ...})
cost.expand(larger_index, positions)  # Broadcast to larger index
```

### `sum_by(expr, positions)`

Aggregate expression by grouping on positions. The index is automatically inferred from Variables/Parameters in the expression.

```python
sum_by(ship, 'warehouses')  # Sum over customers
sum_by(ship, ['warehouses', 'periods'])  # Keep multiple dimensions
sum_by(2 * ship + cost, 'origin')  # Works on expressions too
```

### `where(expr, cond=None, **kwargs)`

Filter expression elements by condition. The index is automatically inferred from Variables/Parameters in the expression.

```python
where(ship, lambda r: r[0] == 'W1')  # Callable filter
where(ship, origin='W1')  # Keyword filter
```

## Examples

See the `examples/` directory for complete examples including multi-period transportation problems.

## License

Apache-2.0
