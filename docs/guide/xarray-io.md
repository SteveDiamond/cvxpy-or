# xarray I/O

cvxpy-or provides functions for loading data from xarray DataArrays and exporting results back to xarray.

## Installation

Install the xarray optional dependency:

```bash
pip install cvxpy-or[xarray]
# or
uv add cvxpy-or[xarray]
```

## Why xarray?

xarray provides labeled, multi-dimensional arrays that are perfect for OR data:

- **Named dimensions** - `warehouse`, `customer`, `period` instead of axis 0, 1, 2
- **Coordinate labels** - `Seattle`, `NYC` instead of indices 0, 1
- **Matrix-style data** - Natural 2D/3D array layout instead of row-by-row dicts
- **Built-in operations** - Slicing, aggregation, and alignment by labels

## Creating Objects from DataArrays

### set_from_dataarray

Create a Set from DataArray coordinates:

```python
import xarray as xr
from cvxpy_or import set_from_dataarray

cost = xr.DataArray(
    [[10, 15], [12, 18]],
    dims=['warehouse', 'customer'],
    coords={'warehouse': ['W1', 'W2'], 'customer': ['C1', 'C2']}
)

routes = set_from_dataarray(cost)
# Set([('W1', 'C1'), ('W1', 'C2'), ('W2', 'C1'), ('W2', 'C2')],
#     names=('warehouse', 'customer'))
```

For 1D DataArrays, creates a simple (non-compound) Set:

```python
supply = xr.DataArray(
    [100, 80, 120],
    dims=['warehouse'],
    coords={'warehouse': ['Seattle', 'Denver', 'Chicago']}
)

warehouses = set_from_dataarray(supply)
# Set(['Seattle', 'Denver', 'Chicago'], name='warehouse')
```

### parameter_from_dataarray

Create Parameters directly from DataArrays:

```python
from cvxpy_or import parameter_from_dataarray

cost_param = parameter_from_dataarray(cost, name='cost')

# Access by tuple key
cost_param.get_value(('W1', 'C2'))  # 15.0
```

### variable_like_dataarray

Create Variables with the same shape as a DataArray:

```python
from cvxpy_or import variable_like_dataarray

ship = variable_like_dataarray(cost, name='ship', nonneg=True)
# Variable indexed by (warehouse, customer) pairs
```

## Aggregation with sum_by

Use the `sum_by` function for aggregation (not xarray's `.sum()` method):

```python
from cvxpy_or import sum_by

# Sum over customers, result indexed by warehouse
total_shipped_from = sum_by(ship, 'warehouse')

# Sum over warehouses, result indexed by customer
total_received_by = sum_by(ship, 'customer')
```

Note: `sum_by(ship, 'warehouse')` means "group by warehouse" (keep warehouse dimension, sum over everything else).

## Exporting to DataArrays

After solving, export results back to xarray:

```python
from cvxpy_or import variable_to_dataarray, parameter_to_dataarray

prob.solve()

# Export solved variable
ship_result = variable_to_dataarray(ship)

# Now use xarray operations for analysis
ship_result.sel(warehouse='Seattle')  # Shipments from Seattle
ship_result.sum(dim='customer')       # Total per warehouse (as xarray)
```

## Complete Example

```python
import cvxpy as cp
import xarray as xr
from cvxpy_or import (
    parameter_from_dataarray,
    variable_like_dataarray,
    variable_to_dataarray,
    sum_by,
)

# Define data as xarray DataArrays
cost = xr.DataArray(
    [[2.5, 1.0, 1.8], [2.0, 1.5, 1.2], [1.0, 2.5, 1.5]],
    dims=['warehouse', 'customer'],
    coords={
        'warehouse': ['Seattle', 'Denver', 'Chicago'],
        'customer': ['NYC', 'LA', 'Houston']
    }
)

supply = xr.DataArray(
    [100, 80, 120],
    dims=['warehouse'],
    coords={'warehouse': ['Seattle', 'Denver', 'Chicago']}
)

demand = xr.DataArray(
    [80, 70, 50],
    dims=['customer'],
    coords={'customer': ['NYC', 'LA', 'Houston']}
)

# Create optimization objects
cost_param = parameter_from_dataarray(cost, name='cost')
supply_param = parameter_from_dataarray(supply, name='supply')
demand_param = parameter_from_dataarray(demand, name='demand')
ship = variable_like_dataarray(cost, name='ship', nonneg=True)

# Build and solve
prob = cp.Problem(
    cp.Minimize(cost_param @ ship),
    [
        sum_by(ship, 'warehouse') <= supply_param,
        sum_by(ship, 'customer') >= demand_param,
    ]
)
prob.solve()

# Export and analyze results
result = variable_to_dataarray(ship)
print(result)
print(f"\nTotal cost: ${prob.value:.2f}")
print(f"\nShipments from Seattle:\n{result.sel(warehouse='Seattle')}")
```

## API Reference

### set_from_dataarray

```python
set_from_dataarray(da: xr.DataArray, name: str | None = None) -> Set
```

Create a Set from DataArray coordinates. For multi-dimensional arrays, creates a compound Set with all coordinate combinations.

### parameter_from_dataarray

```python
parameter_from_dataarray(
    da: xr.DataArray,
    index: Set | None = None,
    name: str | None = None
) -> Parameter
```

Create a Parameter from DataArray values. If `index` is provided, uses that Set; otherwise creates one from coordinates.

### variable_like_dataarray

```python
variable_like_dataarray(
    da: xr.DataArray,
    name: str | None = None,
    nonneg: bool = False,
    **kwargs
) -> Variable
```

Create a Variable with the same shape/coordinates as a DataArray.

### variable_to_dataarray

```python
variable_to_dataarray(var: Variable, name: str | None = None) -> xr.DataArray
```

Convert a solved Variable to a DataArray. Raises `ValueError` if unsolved.

### parameter_to_dataarray

```python
parameter_to_dataarray(param: Parameter, name: str | None = None) -> xr.DataArray
```

Convert a Parameter to a DataArray.

## See Also

- [pandas I/O](pandas-io.md) - For DataFrame-based workflows
- [Quickstart](../quickstart.md) - Basic dict-based approach
