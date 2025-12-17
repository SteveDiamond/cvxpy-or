# Multi-Period Transportation

This example solves a multi-period transportation problem with inventory.

**Source**: `examples/multi_period_transportation.py`

## Problem Description

A company ships products from warehouses to customers over multiple time periods. Each warehouse has supply capacity that varies by period, and each customer has demand that must be met. Unsold supply becomes inventory (with holding cost).

**Objective**: Minimize total shipping + holding cost.

## Mathematical Formulation

**Sets**:
- $W$ = warehouses
- $C$ = customers
- $T$ = time periods

**Parameters**:
- $cost_{w,c}$ = shipping cost per unit from $w$ to $c$
- $supply_{w,t}$ = supply available at warehouse $w$ in period $t$
- $demand_{c,t}$ = demand at customer $c$ in period $t$
- $hold_w$ = holding cost per unit at warehouse $w$

**Variables**:
- $ship_{w,c,t}$ = units shipped from $w$ to $c$ in period $t$
- $inv_{w,t}$ = inventory at warehouse $w$ at end of period $t$

**Constraints**:
- Supply: $\sum_c ship_{w,c,t} \leq supply_{w,t}$ for all $w, t$
- Demand: $\sum_w ship_{w,c,t} \geq demand_{c,t}$ for all $c, t$
- Inventory: $inv_{w,t} = supply_{w,t} - \sum_c ship_{w,c,t}$

**Objective**: Minimize $\sum_{w,c,t} cost_{w,c} \cdot ship_{w,c,t} + \sum_{w,t} hold_w \cdot inv_{w,t}$

## Implementation

### Data Setup

```python
import pandas as pd
from cvxpy_or import (
    Model, Set, sum_by,
    set_from_dataframe, parameter_from_dataframe, parameter_from_series,
    variable_to_dataframe, print_variable,
)

# Shipping cost per unit (warehouse -> customer)
cost_df = pd.DataFrame([
    {'warehouse': 'Seattle', 'customer': 'NYC', 'cost': 2.5},
    {'warehouse': 'Seattle', 'customer': 'LA', 'cost': 1.0},
    {'warehouse': 'Denver', 'customer': 'NYC', 'cost': 2.0},
    {'warehouse': 'Denver', 'customer': 'LA', 'cost': 1.5},
    # ... more routes
])

# Supply capacity per warehouse per period
supply_df = pd.DataFrame([
    {'warehouse': 'Seattle', 'period': 'Jan', 'supply': 100},
    {'warehouse': 'Seattle', 'period': 'Feb', 'supply': 120},
    {'warehouse': 'Denver', 'period': 'Jan', 'supply': 80},
    # ... more supply data
])

# Customer demand per period
demand_df = pd.DataFrame([
    {'customer': 'NYC', 'period': 'Jan', 'demand': 60},
    {'customer': 'NYC', 'period': 'Feb', 'demand': 70},
    # ... more demand data
])
```

### Model Definition

```python
m = Model(name='multi_period_transport')

# Define sets
warehouses = Set(cost_df['warehouse'].unique().tolist(), name='warehouses')
customers = Set(cost_df['customer'].unique().tolist(), name='customers')
periods = Set(['Jan', 'Feb', 'Mar'], name='periods')

# Cross-product indices
routes = Set.cross(warehouses, customers, name='routes')
shipments = Set.cross(warehouses, customers, periods, name='shipments')
inventory_idx = Set.cross(warehouses, periods, name='inventory_idx')
```

### Parameters

```python
cost = parameter_from_dataframe(cost_df, ['warehouse', 'customer'], 'cost', name='cost')
supply = parameter_from_dataframe(supply_df, ['warehouse', 'period'], 'supply', name='supply')
demand = parameter_from_dataframe(demand_df, ['customer', 'period'], 'demand', name='demand')
holding_cost = parameter_from_series(holding_cost_series, name='holding_cost')
```

### Variables

```python
ship = m.add_variable(shipments, nonneg=True, name='ship')
inv = m.add_variable(inventory_idx, nonneg=True, name='inventory')
```

### Constraints

```python
# Supply constraint: shipped from each warehouse/period <= supply
m.add_constraint('supply', sum_by(ship, ['warehouses', 'periods']) <= supply)

# Demand constraint: received by each customer/period >= demand
m.add_constraint('demand', sum_by(ship, ['customers', 'periods']) >= demand)

# Inventory balance
m.add_constraint('inv_balance', inv == supply - sum_by(ship, ['warehouses', 'periods']))
```

### Objective

```python
shipping_cost = cost @ sum_by(ship, ['warehouses', 'customers'])
holding_cost_expr = holding_cost @ sum_by(inv, 'warehouses')
m.minimize(shipping_cost + holding_cost_expr)
```

### Solve and Analyze

```python
m.solve()
m.print_summary()

# Export to DataFrame for analysis
ship_df = variable_to_dataframe(ship, value_col='quantity')
ship_df = ship_df[ship_df['quantity'] > 0.01]

# Pivot by period
for period in periods:
    period_df = ship_df[ship_df['periods'] == period]
    pivot = period_df.pivot(index='warehouses', columns='customers', values='quantity')
    print(f"\n{period}:\n{pivot}")
```

## Key Patterns

### Multi-dimensional sum_by

The `sum_by` function handles multi-dimensional indices naturally:

```python
# shipments has index (warehouses, customers, periods)

# Sum over customers, keep warehouses and periods
sum_by(ship, ['warehouses', 'periods'])

# Sum over warehouses and periods, keep customers
sum_by(ship, 'customers')

# Sum over warehouses and customers, keep periods
sum_by(ship, 'periods')
```

### Computing Aggregate Costs

```python
# Shipping cost: cost @ (sum over periods)
shipping_cost = cost @ sum_by(ship, ['warehouses', 'customers'])

# Holding cost: holding_cost @ (sum over periods)
holding_cost_expr = holding_cost @ sum_by(inv, 'warehouses')
```

### DataFrame Analysis

```python
# Export and pivot for visualization
ship_df = variable_to_dataframe(ship)
pivot = ship_df.pivot_table(
    index='warehouses',
    columns=['periods', 'customers'],
    values='value',
    fill_value=0
)
```
