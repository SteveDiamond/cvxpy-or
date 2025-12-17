# Quickstart

This guide walks through building a classic transportation problem with cvxpy-or.

## The Problem

A company has warehouses in Seattle, Denver, and Chicago that need to ship products to customers in NYC, LA, and Houston. Each warehouse has limited supply, each customer has demand that must be met, and shipping costs vary by route. **Goal**: Minimize total shipping cost.

## Step 1: Import and Define Data

cvxpy-or uses pandas DataFrames as the primary data format:

```python
import pandas as pd
from cvxpy_or import (
    Model, Set, sum_by,
    set_from_dataframe, parameter_from_dataframe, parameter_from_series,
)

# Shipping costs as a DataFrame (long format)
cost_df = pd.DataFrame([
    {"warehouse": "Seattle", "customer": "NYC", "cost": 2.5},
    {"warehouse": "Seattle", "customer": "LA", "cost": 1.0},
    {"warehouse": "Seattle", "customer": "Houston", "cost": 1.8},
    {"warehouse": "Denver", "customer": "NYC", "cost": 2.0},
    {"warehouse": "Denver", "customer": "LA", "cost": 1.5},
    {"warehouse": "Denver", "customer": "Houston", "cost": 1.2},
    {"warehouse": "Chicago", "customer": "NYC", "cost": 1.0},
    {"warehouse": "Chicago", "customer": "LA", "cost": 2.5},
    {"warehouse": "Chicago", "customer": "Houston", "cost": 1.5},
])

# Supply and demand as Series
supply = pd.Series(
    {"Seattle": 100, "Denver": 80, "Chicago": 120},
    name="supply"
)
demand = pd.Series(
    {"NYC": 80, "LA": 70, "Houston": 50},
    name="demand"
)
```

## Step 2: Create Sets and Parameters from Data

```python
# Create route index from the cost table
routes = set_from_dataframe(cost_df, ["warehouse", "customer"])

# Load parameters directly from DataFrames/Series
cost = parameter_from_dataframe(cost_df, ["warehouse", "customer"], "cost", name="cost")
supply_param = parameter_from_series(supply, name="supply")
demand_param = parameter_from_series(demand, name="demand")
```

## Step 3: Build the Model

```python
m = Model(name="transportation")

# Decision variable: how much to ship on each route
ship = m.add_variable(routes, nonneg=True, name="ship")

# Supply constraint: total shipped from each warehouse <= supply
m.add_constraint("supply", sum_by(ship, "warehouse") <= supply_param)

# Demand constraint: total received by each customer >= demand
m.add_constraint("demand", sum_by(ship, "customer") >= demand_param)

# Objective: minimize total shipping cost
m.minimize(cost @ ship)
```

## Step 4: Solve and View Results

```python
m.solve()
m.print_summary()
m.print_solution(show_zero=False)

# Export results back to DataFrame
result_df = m.to_dataframe("ship")
print(result_df[result_df["value"] > 0])
```

## Complete Code

```python
import pandas as pd
from cvxpy_or import (
    Model, Set, sum_by,
    set_from_dataframe, parameter_from_dataframe, parameter_from_series,
)

# Define data as DataFrames
cost_df = pd.DataFrame([
    {"warehouse": "Seattle", "customer": "NYC", "cost": 2.5},
    {"warehouse": "Seattle", "customer": "LA", "cost": 1.0},
    {"warehouse": "Seattle", "customer": "Houston", "cost": 1.8},
    {"warehouse": "Denver", "customer": "NYC", "cost": 2.0},
    {"warehouse": "Denver", "customer": "LA", "cost": 1.5},
    {"warehouse": "Denver", "customer": "Houston", "cost": 1.2},
    {"warehouse": "Chicago", "customer": "NYC", "cost": 1.0},
    {"warehouse": "Chicago", "customer": "LA", "cost": 2.5},
    {"warehouse": "Chicago", "customer": "Houston", "cost": 1.5},
])

supply = pd.Series({"Seattle": 100, "Denver": 80, "Chicago": 120}, name="supply")
demand = pd.Series({"NYC": 80, "LA": 70, "Houston": 50}, name="demand")

# Build model from DataFrames
routes = set_from_dataframe(cost_df, ["warehouse", "customer"])
cost = parameter_from_dataframe(cost_df, ["warehouse", "customer"], "cost", name="cost")
supply_param = parameter_from_series(supply, name="supply")
demand_param = parameter_from_series(demand, name="demand")

m = Model(name="transportation")
ship = m.add_variable(routes, nonneg=True, name="ship")

# Constraints
m.add_constraint("supply", sum_by(ship, "warehouse") <= supply_param)
m.add_constraint("demand", sum_by(ship, "customer") >= demand_param)

# Solve
m.minimize(cost @ ship)
m.solve()

# Display results
m.print_summary()
m.print_solution(show_zero=False)

# Export to DataFrame for further analysis
result_df = m.to_dataframe("ship")
result_df = result_df[result_df["value"] > 0]
result_df = result_df.merge(cost_df, on=["warehouse", "customer"])
result_df["shipping_cost"] = result_df["value"] * result_df["cost"]
print(result_df)
```

## What's Next?

- Learn about [Sets, Variables, and Parameters](guide/basic-usage.md)
- Explore [aggregation functions](guide/aggregations.md) like `mean_by`, `min_by`, `max_by`
- See [constraint helpers](guide/constraints.md) for cardinality and logical constraints
- Try [xarray I/O](guide/xarray-io.md) for matrix-style data
- Browse [more examples](examples/index.md) including assignment, facility location, and more
