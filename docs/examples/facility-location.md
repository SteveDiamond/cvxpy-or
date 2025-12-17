# Facility Location

This example solves the uncapacitated facility location problem (UFLP).

**Source**: `examples/facility_location.py`

## Problem Description

A company must decide which facilities to open from a set of potential locations, and how to assign customers to open facilities. Opening a facility has a fixed cost, and serving customers from a facility has transportation costs.

**Objective**: Minimize total fixed cost + transportation cost.

## Mathematical Formulation

**Sets**:
- $F$ = potential facility locations
- $C$ = customers

**Parameters**:
- $fixed_f$ = fixed cost to open facility $f$
- $capacity_f$ = capacity of facility $f$
- $transport_{f,c}$ = cost per unit to serve customer $c$ from facility $f$
- $demand_c$ = demand of customer $c$

**Variables**:
- $open_f \in [0,1]$ = fraction of facility $f$ that is open
- $ship_{f,c} \geq 0$ = units shipped from $f$ to $c$

**Constraints**:
- Demand satisfaction: $\sum_f ship_{f,c} \geq demand_c$ for all $c$
- Capacity linking: $\sum_c ship_{f,c} \leq capacity_f \cdot open_f$ for all $f$
- Open bounds: $open_f \leq 1$

**Objective**: Minimize $\sum_f fixed_f \cdot open_f + \sum_{f,c} transport_{f,c} \cdot ship_{f,c}$

## Implementation

### Data Setup

```python
import pandas as pd
import cvxpy as cp
from cvxpy_or import (
    Model, Set, sum_by, validate_keys, ValidationError,
    parameter_from_dataframe, parameter_from_series,
    variable_to_dataframe, print_variable,
)

# Facility data
facility_df = pd.DataFrame([
    {'facility': 'Atlanta', 'fixed_cost': 500, 'capacity': 1000, 'region': 'South'},
    {'facility': 'Boston', 'fixed_cost': 600, 'capacity': 1000, 'region': 'Northeast'},
    {'facility': 'Chicago', 'fixed_cost': 550, 'capacity': 1000, 'region': 'Midwest'},
    {'facility': 'Denver', 'fixed_cost': 450, 'capacity': 1000, 'region': 'West'},
    {'facility': 'Seattle', 'fixed_cost': 650, 'capacity': 1000, 'region': 'West'},
])

# Customer data
customer_df = pd.DataFrame([
    {'customer': 'NYC', 'demand': 100, 'region': 'Northeast'},
    {'customer': 'LA', 'demand': 150, 'region': 'West'},
    {'customer': 'Houston', 'demand': 80, 'region': 'South'},
    {'customer': 'Phoenix', 'demand': 60, 'region': 'West'},
])

# Transportation costs
transport_df = pd.DataFrame([
    {'facility': 'Atlanta', 'customer': 'NYC', 'cost': 15},
    {'facility': 'Atlanta', 'customer': 'LA', 'cost': 40},
    # ... more routes
])
```

### Validation

```python
# Validate data matches expected indices
fixed_cost_data = facility_df.set_index('facility')['fixed_cost'].to_dict()

try:
    validate_keys(fixed_cost_data, facilities)
    print("Fixed cost data validated successfully")
except ValidationError as e:
    print(f"Validation error: {e}")
```

### Model Definition

```python
m = Model(name='facility_location')

# Define sets
facilities = Set(facility_df['facility'].tolist(), name='facilities')
customers = Set(customer_df['customer'].tolist(), name='customers')
connections = Set.cross(facilities, customers, name='connections')

# Parameters
fixed_cost = parameter_from_series(
    facility_df.set_index('facility')['fixed_cost'], name='fixed_cost')
capacity = parameter_from_series(
    facility_df.set_index('facility')['capacity'], name='capacity')
transport_cost = parameter_from_dataframe(
    transport_df, ['facility', 'customer'], 'cost', name='transport_cost')
demand = parameter_from_series(
    customer_df.set_index('customer')['demand'], name='demand')
```

### Variables

```python
# Binary (relaxed): 1 if facility is opened
open_facility = m.add_variable(facilities, nonneg=True, name='open')

# Amount shipped from facility to customer
ship = m.add_variable(connections, nonneg=True, name='ship')
```

### Constraints

```python
# Demand satisfaction
m.add_constraint('demand', sum_by(ship, 'customers') >= demand)

# Capacity linking: can only ship from open facilities
m.add_constraint('capacity',
    sum_by(ship, 'facilities') <= cp.multiply(capacity, open_facility))

# Facility open variable bounds
m.add_constraint('open_bound', open_facility <= 1)
```

### Objective

```python
fixed_cost_expr = fixed_cost @ open_facility
transport_cost_expr = transport_cost @ ship
m.minimize(fixed_cost_expr + transport_cost_expr)
```

### Solve and Analyze

```python
m.solve()

print(f"Fixed cost:     ${fixed_cost_expr.value:,.0f}k")
print(f"Transport cost: ${transport_cost_expr.value:,.0f}k")
print(f"Total cost:     ${m.value:,.0f}k")

# Facility decisions
facility_results = variable_to_dataframe(open_facility, value_col='open_level')
facility_results['status'] = facility_results['open_level'].apply(
    lambda x: 'OPEN' if x > 0.99 else ('PARTIAL' if x > 0.01 else 'CLOSED'))
print(facility_results)
```

## Key Patterns

### Linking Constraints

The key constraint links shipping to facility opening:

```python
# Can only ship from open facilities
# ship[f,c] <= capacity[f] * open[f]
m.add_constraint('capacity',
    sum_by(ship, 'facilities') <= cp.multiply(capacity, open_facility))
```

### LP Relaxation vs Integer

The LP relaxation may give fractional `open_facility` values. For integer solutions:

```python
# Integer version (slower but exact)
open_facility = m.add_variable(facilities, boolean=True, name='open')
```

### Cost Breakdown

```python
# Separate fixed and variable costs
fixed_cost_expr = fixed_cost @ open_facility
transport_cost_expr = transport_cost @ ship

# Analyze
print(f"Fixed: ${fixed_cost_expr.value:,.0f}")
print(f"Variable: ${transport_cost_expr.value:,.0f}")
```

### Limiting Open Facilities

```python
from cvxpy_or import at_most_k

# Open at most 3 facilities
constraints = at_most_k(open_facility, k=3)
m.add_constraint('max_facilities', constraints)
```

## Variations

### Capacitated Facility Location

```python
# Stricter capacity constraint
m.add_constraint('capacity',
    sum_by(ship, 'facilities') <= capacity * open_facility)
```

### Single-Source Assignment

Each customer served by exactly one facility:

```python
# Binary assignment
assign = m.add_variable(connections, boolean=True, name='assign')
m.add_constraint('single_source', sum_by(assign, 'customers') == 1)

# Link shipping to assignment
m.add_constraint('link', ship <= demand.expand(connections, 'customers') * assign)
```

### Minimum Facilities

```python
from cvxpy_or import at_least_k

# Open at least 2 facilities
constraints = at_least_k(open_facility, k=2)
m.add_constraint('min_facilities', constraints)
```
