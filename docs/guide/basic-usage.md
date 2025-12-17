# Basic Usage

This guide covers the core classes: Set, Variable, Parameter, and Model.

## Sets

A `Set` is an ordered collection of elements that serves as an index for variables and parameters.

### Creating Sets

```python
from cvxpy_or import Set

# Simple set
cities = Set(['NYC', 'LA', 'Chicago'], name='cities')

# Set from any iterable
numbers = Set(range(1, 6), name='numbers')  # {1, 2, 3, 4, 5}
```

### Compound Sets (Cross Products)

For multi-dimensional indexing, create compound sets:

```python
origins = Set(['Seattle', 'Denver'], name='origins')
destinations = Set(['NYC', 'LA'], name='destinations')

# Cross product creates all pairs
routes = Set.cross(origins, destinations)
# Contains: ('Seattle', 'NYC'), ('Seattle', 'LA'), ('Denver', 'NYC'), ('Denver', 'LA')

# Access the component names
print(routes.names)  # ('origins', 'destinations')
```

### Set Operations

```python
A = Set([1, 2, 3])
B = Set([2, 3, 4])

A | B  # Union: {1, 2, 3, 4}
A & B  # Intersection: {2, 3}
A - B  # Difference: {1}
A ^ B  # Symmetric difference: {1, 4}
A <= B # Subset check: False
```

### Filtering and Transformation

```python
numbers = Set(range(1, 11), name='numbers')

# Filter elements
evens = numbers.filter(lambda x: x % 2 == 0)  # {2, 4, 6, 8, 10}

# Transform elements
doubled = numbers.map(lambda x: x * 2)  # {2, 4, 6, ..., 20}

# Sort
sorted_set = numbers.sorted(reverse=True)  # {10, 9, 8, ..., 1}
```

### Set Properties

```python
s = Set(['a', 'b', 'c'], name='items')

len(s)        # 3
'a' in s      # True
s.first()     # 'a'
s.last()      # 'c'
list(s)       # ['a', 'b', 'c']
```

## Variables

A `Variable` is a CVXPY Variable indexed by a Set.

### Creating Variables

```python
from cvxpy_or import Variable, Set

items = Set(['A', 'B', 'C'], name='items')

# Basic variable
x = Variable(items, name='x')

# Non-negative variable
x = Variable(items, nonneg=True, name='x')

# Integer variable
x = Variable(items, integer=True, name='x')

# Boolean variable
x = Variable(items, boolean=True, name='x')
```

### Accessing Elements

```python
routes = Set.cross(
    Set(['W1', 'W2'], name='warehouses'),
    Set(['C1', 'C2'], name='customers')
)
ship = Variable(routes, nonneg=True, name='ship')

# Access by key (returns CVXPY expression for constraints)
ship[('W1', 'C1')]

# Get solved value (after solving)
ship.get_value(('W1', 'C1'))
```

### CVXPY Compatibility

Variables inherit from CVXPY, so all operations work:

```python
import cvxpy as cp

x = Variable(items, nonneg=True)
y = Variable(items, nonneg=True)

# Arithmetic
x + y
x - y
2 * x
x / 2

# CVXPY functions
cp.sum(x)
cp.norm(x)
cp.max(x)

# Constraints
x <= 10
x >= 0
x == y
```

## Parameters

A `Parameter` is a CVXPY Parameter indexed by a Set.

### Creating Parameters

```python
from cvxpy_or import Parameter, Set

items = Set(['A', 'B', 'C'], name='items')

# With data
cost = Parameter(items, data={'A': 10, 'B': 20, 'C': 15}, name='cost')

# Non-negative parameter
cost = Parameter(items, nonneg=True, data={'A': 10, 'B': 20, 'C': 15})
```

### Updating Values

```python
# Update all values
cost.set_data({'A': 15, 'B': 25, 'C': 20})

# Re-solve with new data (no need to rebuild the problem)
m.solve()
```

### Parameter Expansion

Expand a parameter to a larger index:

```python
warehouses = Set(['W1', 'W2'], name='warehouses')
routes = Set.cross(warehouses, Set(['C1', 'C2'], name='customers'))

# Capacity per warehouse
capacity = Parameter(warehouses, data={'W1': 100, 'W2': 80})

# Expand to routes (broadcast)
capacity_per_route = capacity.expand(routes, 'warehouses')
# Each route gets the capacity of its warehouse
```

## Model

The `Model` class provides a clean interface for building optimization problems.

### Creating a Model

```python
from cvxpy_or import Model

m = Model(name='my_problem')
```

### Adding Components

```python
# Add parameter (returns Parameter)
cost = m.add_parameter(items, data={...}, name='cost')

# Add variable (returns Variable)
x = m.add_variable(items, nonneg=True, name='x')

# Add constraint
m.add_constraint('budget', cost @ x <= 100)
m.add_constraint('limits', x <= 10)

# Set objective
m.minimize(cost @ x)
# or
m.maximize(profit @ x)
```

### Solving

```python
status = m.solve()
print(status)  # 'optimal', 'infeasible', etc.
```

Pass solver options:

```python
m.solve(solver='CLARABEL', verbose=True)
```

### Accessing Results

```python
# Objective value
print(m.value)

# Status
print(m.status)

# Get a variable by name
x = m.get_variable('x')

# Get a parameter by name
cost = m.get_parameter('cost')

# List all variables/parameters
print(m.variables)
print(m.parameters)
```

### Displaying Results

```python
# Summary of model and solution
m.print_summary()

# Solution values for all variables
m.print_solution()

# Hide zero values
m.print_solution(show_zero=False)

# Control precision
m.print_solution(precision=4)
```

### Exporting to DataFrame

```python
# Export a specific variable
df = m.to_dataframe('x')

# Contains columns for each index dimension plus 'value'
```

## Complete Example

```python
from cvxpy_or import Model, Set, sum_by

# Model
m = Model(name='production')

# Sets
products = Set(['Widget', 'Gadget'], name='products')
resources = Set(['Labor', 'Material'], name='resources')

# Parameters
profit = m.add_parameter(products, name='profit',
                         data={'Widget': 30, 'Gadget': 50})
available = m.add_parameter(resources, name='available',
                            data={'Labor': 40, 'Material': 60})

# Resource usage (products x resources)
usage_index = Set.cross(products, resources)
usage = m.add_parameter(usage_index, name='usage', data={
    ('Widget', 'Labor'): 1, ('Widget', 'Material'): 2,
    ('Gadget', 'Labor'): 2, ('Gadget', 'Material'): 3,
})

# Decision variable
produce = m.add_variable(products, nonneg=True, name='produce')

# Constraints: resource limits
for r in resources:
    resource_usage = sum_by(
        usage * produce.expand(usage_index, 'products'),
        'resources'
    )
m.add_constraint('resources', sum_by(usage * produce.expand(usage_index, 'products'), 'products') <= available)

# Objective: maximize profit
m.maximize(profit @ produce)

# Solve
m.solve()
m.print_summary()
m.print_solution()
```
