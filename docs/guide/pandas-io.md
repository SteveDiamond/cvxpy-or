# pandas Integration

cvxpy-or integrates seamlessly with pandas for loading data and exporting results.

## Creating Sets from DataFrames

### set_from_series

Create a Set from unique values in a Series:

```python
import pandas as pd
from cvxpy_or import set_from_series

df = pd.DataFrame({
    'customer': ['Alice', 'Bob', 'Alice', 'Carol'],
    'product': ['Widget', 'Gadget', 'Widget', 'Widget'],
    'quantity': [10, 20, 15, 5]
})

customers = set_from_series(df['customer'])
# Set(['Alice', 'Bob', 'Carol'])

products = set_from_series(df['product'])
# Set(['Widget', 'Gadget'])
```

### set_from_dataframe

Create a compound Set from multiple columns:

```python
from cvxpy_or import set_from_dataframe

# Create Set of (customer, product) pairs
orders = set_from_dataframe(df, columns=['customer', 'product'])
# Set with tuples: ('Alice', 'Widget'), ('Bob', 'Gadget'), ...
```

### set_from_index

Create a Set from a DataFrame's index:

```python
from cvxpy_or import set_from_index

indexed_df = df.set_index('customer')
customers = set_from_index(indexed_df)
```

For MultiIndex:

```python
multi_indexed = df.set_index(['customer', 'product'])
customer_products = set_from_index(multi_indexed)
```

## Creating Parameters from DataFrames

### parameter_from_dataframe

Load a Parameter from DataFrame columns:

```python
from cvxpy_or import parameter_from_dataframe

cost_df = pd.DataFrame({
    'origin': ['Seattle', 'Seattle', 'Denver', 'Denver'],
    'destination': ['NYC', 'LA', 'NYC', 'LA'],
    'cost': [2.5, 1.0, 2.0, 1.5]
})

cost = parameter_from_dataframe(
    cost_df,
    index_cols=['origin', 'destination'],
    value_col='cost',
    name='cost'
)
```

### parameter_from_series

Load a Parameter from a Series with an index:

```python
from cvxpy_or import parameter_from_series

supply_series = pd.Series(
    {'Seattle': 100, 'Denver': 80, 'Chicago': 120},
    name='supply'
)

supply = parameter_from_series(supply_series, name='supply')
```

From a DataFrame column:

```python
facility_df = pd.DataFrame({
    'facility': ['Seattle', 'Denver', 'Chicago'],
    'capacity': [100, 80, 120]
})

capacity = parameter_from_series(
    facility_df.set_index('facility')['capacity'],
    name='capacity'
)
```

## Exporting Results to DataFrames

### variable_to_dataframe

Export a Variable's values after solving:

```python
from cvxpy_or import variable_to_dataframe

# After solving
df = variable_to_dataframe(ship, value_col='shipped')
```

Result:
```
   warehouses  customers  shipped
0     Seattle        NYC     30.0
1     Seattle         LA     70.0
2      Denver        NYC     50.0
...
```

### parameter_to_dataframe

Export a Parameter's values:

```python
from cvxpy_or import parameter_to_dataframe

df = parameter_to_dataframe(cost, value_col='cost')
```

### Model.to_dataframe

Export directly from the Model:

```python
# Export a specific variable
ship_df = m.to_dataframe('ship')

# Filter to non-zero values
ship_df = ship_df[ship_df['value'] > 0.01]
```

## Complete Example

```python
import pandas as pd
from cvxpy_or import (
    Model, Set, sum_by,
    set_from_series, set_from_dataframe,
    parameter_from_dataframe, parameter_from_series,
    variable_to_dataframe
)

# Load data from CSV (simulated here with DataFrames)
cost_df = pd.DataFrame({
    'origin': ['Seattle', 'Seattle', 'Denver', 'Denver'],
    'dest': ['NYC', 'LA', 'NYC', 'LA'],
    'cost': [2.5, 1.0, 2.0, 1.5]
})

supply_df = pd.DataFrame({
    'warehouse': ['Seattle', 'Denver'],
    'supply': [100, 80]
})

demand_df = pd.DataFrame({
    'customer': ['NYC', 'LA'],
    'demand': [80, 70]
})

# Create model
m = Model(name='transportation')

# Create sets from data
origins = set_from_series(cost_df['origin'], name='origins')
dests = set_from_series(cost_df['dest'], name='dests')
routes = set_from_dataframe(cost_df, columns=['origin', 'dest'])

# Load parameters from DataFrames
cost = parameter_from_dataframe(
    cost_df, index_cols=['origin', 'dest'], value_col='cost', name='cost'
)
supply = parameter_from_series(
    supply_df.set_index('warehouse')['supply'], name='supply'
)
demand = parameter_from_series(
    demand_df.set_index('customer')['demand'], name='demand'
)

# Create variable
ship = m.add_variable(routes, nonneg=True, name='ship')

# Constraints
m.add_constraint('supply', sum_by(ship, 'origins') <= supply)
m.add_constraint('demand', sum_by(ship, 'dests') >= demand)

# Objective
m.minimize(cost @ ship)

# Solve
m.solve()

# Export results
result_df = variable_to_dataframe(ship, value_col='shipped')
result_df = result_df[result_df['shipped'] > 0.01]

# Merge with original cost data
result_df = result_df.merge(
    cost_df,
    left_on=['origins', 'dests'],
    right_on=['origin', 'dest']
)
result_df['shipping_cost'] = result_df['shipped'] * result_df['cost']

print(result_df[['origin', 'dest', 'shipped', 'cost', 'shipping_cost']])
```

## Tips

### Handling Missing Data

```python
# Fill missing values before creating parameters
df['cost'] = df['cost'].fillna(999)  # High cost for missing routes
```

### Filtering Data

```python
# Only include active routes
active_routes = cost_df[cost_df['active'] == True]
routes = set_from_dataframe(active_routes, columns=['origin', 'dest'])
```

### Pivoting Results

```python
# Pivot to matrix format
result_df = variable_to_dataframe(ship)
pivot = result_df.pivot(index='origins', columns='dests', values='value')
print(pivot)
```

### Joining with Original Data

```python
# Add metadata to results
result_df = variable_to_dataframe(ship)
result_df = result_df.merge(facility_df, left_on='facilities', right_on='facility')
```
