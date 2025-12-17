# Quickstart (xarray)

This guide shows the same transportation problem using xarray DataArrays instead of pandas.

xarray is ideal when your data is naturally matrix-shaped (e.g., cost matrices, distance tables) and you want labeled coordinates.

## Installation

xarray is an optional dependency:

```bash
pip install cvxpy-or[xarray]
# or
uv add cvxpy-or[xarray]
```

## The Problem

Same as the pandas quickstart: minimize shipping costs from warehouses to customers.

## Step 1: Define Data as DataArrays

```python
import xarray as xr
from cvxpy_or import (
    parameter_from_dataarray,
    variable_like_dataarray,
    variable_to_dataarray,
    sum_by,
)
import cvxpy as cp

# Shipping costs as a 2D array with labeled coordinates
cost = xr.DataArray(
    [[2.5, 1.0, 1.8],   # Seattle -> NYC, LA, Houston
     [2.0, 1.5, 1.2],   # Denver  -> NYC, LA, Houston
     [1.0, 2.5, 1.5]],  # Chicago -> NYC, LA, Houston
    dims=["warehouse", "customer"],
    coords={
        "warehouse": ["Seattle", "Denver", "Chicago"],
        "customer": ["NYC", "LA", "Houston"],
    },
    name="cost",
)

# Supply and demand as 1D arrays
supply = xr.DataArray(
    [100, 80, 120],
    dims=["warehouse"],
    coords={"warehouse": ["Seattle", "Denver", "Chicago"]},
    name="supply",
)

demand = xr.DataArray(
    [80, 70, 50],
    dims=["customer"],
    coords={"customer": ["NYC", "LA", "Houston"]},
    name="demand",
)
```

## Step 2: Create Parameters and Variables

```python
# Parameters directly from DataArrays
cost_param = parameter_from_dataarray(cost, name="cost")
supply_param = parameter_from_dataarray(supply, name="supply")
demand_param = parameter_from_dataarray(demand, name="demand")

# Variable has same shape as cost (warehouse x customer)
ship = variable_like_dataarray(cost, name="ship", nonneg=True)
```

## Step 3: Build and Solve

```python
prob = cp.Problem(
    cp.Minimize(cost_param @ ship),
    [
        sum_by(ship, "warehouse") <= supply_param,  # Supply constraint
        sum_by(ship, "customer") >= demand_param,   # Demand constraint
    ],
)

prob.solve()
print(f"Status: {prob.status}")
print(f"Optimal cost: ${prob.value:.2f}")
```

## Step 4: Get Results as DataArray

```python
# Export solution back to xarray
result = variable_to_dataarray(ship)
print(result)

# Use xarray's selection and aggregation
print("\nShipments from Seattle:")
print(result.sel(warehouse="Seattle"))

print("\nTotal shipped to each customer:")
print(result.sum(dim="warehouse"))

print("\nTotal shipped from each warehouse:")
print(result.sum(dim="customer"))
```

## Complete Code

```python
import xarray as xr
import cvxpy as cp
from cvxpy_or import (
    parameter_from_dataarray,
    variable_like_dataarray,
    variable_to_dataarray,
    sum_by,
)

# Define data as DataArrays
cost = xr.DataArray(
    [[2.5, 1.0, 1.8],
     [2.0, 1.5, 1.2],
     [1.0, 2.5, 1.5]],
    dims=["warehouse", "customer"],
    coords={
        "warehouse": ["Seattle", "Denver", "Chicago"],
        "customer": ["NYC", "LA", "Houston"],
    },
    name="cost",
)

supply = xr.DataArray(
    [100, 80, 120],
    dims=["warehouse"],
    coords={"warehouse": ["Seattle", "Denver", "Chicago"]},
    name="supply",
)

demand = xr.DataArray(
    [80, 70, 50],
    dims=["customer"],
    coords={"customer": ["NYC", "LA", "Houston"]},
    name="demand",
)

# Create optimization objects
cost_param = parameter_from_dataarray(cost, name="cost")
supply_param = parameter_from_dataarray(supply, name="supply")
demand_param = parameter_from_dataarray(demand, name="demand")
ship = variable_like_dataarray(cost, name="ship", nonneg=True)

# Build and solve
prob = cp.Problem(
    cp.Minimize(cost_param @ ship),
    [
        sum_by(ship, "warehouse") <= supply_param,
        sum_by(ship, "customer") >= demand_param,
    ],
)

prob.solve()
print(f"Status: {prob.status}")
print(f"Optimal cost: ${prob.value:.2f}")

# Get results
result = variable_to_dataarray(ship)
print("\nOptimal shipments:")
print(result)
```

## When to Use xarray vs pandas

| Use Case | Recommended |
|----------|-------------|
| Data from CSV/database | pandas |
| Sparse data (not all routes exist) | pandas |
| Matrix/grid data | xarray |
| Multi-dimensional arrays | xarray |
| Scientific computing workflow | xarray |
| Joining results with other tables | pandas |

## What's Next?

- See the [main quickstart](quickstart.md) for the pandas approach
- Learn more about [xarray I/O](guide/xarray-io.md)
- Browse [examples](examples/index.md)
