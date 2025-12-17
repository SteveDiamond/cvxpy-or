"""xarray I/O utilities for cvxpy-or.

This module provides functions for loading Sets and Parameters from
xarray DataArrays and exporting Variable values back to DataArrays.
"""

from __future__ import annotations

from itertools import product
from typing import TYPE_CHECKING, Any, cast

import numpy as np

if TYPE_CHECKING:
    import xarray as xr

    from cvxpy_or.sets import Parameter, Set, Variable


def _check_xarray():
    """Check that xarray is available."""
    try:
        import xarray  # noqa: F401

        return xarray
    except ImportError as err:
        raise ImportError(
            "xarray is required for xarray I/O operations. Install it with: uv add xarray"
        ) from err


def set_from_dataarray(
    da: xr.DataArray,
    name: str | None = None,
) -> Set:
    """Create a Set from a DataArray's coordinates.

    For a multi-dimensional DataArray, creates a compound Set containing
    tuples of all coordinate combinations (Cartesian product of dims).

    Parameters
    ----------
    da : xr.DataArray
        The DataArray whose coordinates define the Set elements.
    name : str, optional
        Name for the Set. Defaults to DataArray name.

    Returns
    -------
    Set
        A new Set indexed by the DataArray coordinates.

    Examples
    --------
    >>> import xarray as xr
    >>> cost = xr.DataArray(
    ...     [[10, 15], [12, 18]],
    ...     dims=['warehouse', 'customer'],
    ...     coords={'warehouse': ['W1', 'W2'], 'customer': ['C1', 'C2']}
    ... )
    >>> routes = set_from_dataarray(cost)
    >>> list(routes)
    [('W1', 'C1'), ('W1', 'C2'), ('W2', 'C1'), ('W2', 'C2')]
    """
    _check_xarray()
    from cvxpy_or.sets import Set as SetClass

    if len(da.dims) == 0:
        raise ValueError("DataArray must have at least one dimension")

    if len(da.dims) == 1:
        # Single dimension - simple Set
        dim_name = str(da.dims[0])
        elements = list(da.coords[dim_name].values)
        set_name = name or (str(da.name) if da.name is not None else dim_name)
        return SetClass(elements, name=set_name)

    # Multi-dimensional - compound Set
    # Generate all combinations in C-order (row-major) to match DataArray flattening
    coord_arrays = [list(da.coords[dim].values) for dim in da.dims]
    elements = list(product(*coord_arrays))
    dim_names = tuple(str(d) for d in da.dims)

    set_name = name or (str(da.name) if da.name is not None else None)
    return SetClass(elements, name=set_name, names=dim_names)


def parameter_from_dataarray(
    da: xr.DataArray,
    index: Set | None = None,
    name: str | None = None,
) -> Parameter:
    """Create a Parameter from a DataArray.

    The DataArray's coordinates define the index, and values become
    the parameter data.

    Parameters
    ----------
    da : xr.DataArray
        The DataArray containing parameter values.
    index : Set, optional
        Existing Set to use. If not provided, creates from DataArray coords.
    name : str, optional
        Name for the Parameter.

    Returns
    -------
    Parameter
        A new Parameter with data from the DataArray.

    Examples
    --------
    >>> import xarray as xr
    >>> cost = xr.DataArray(
    ...     [[10, 15], [12, 18]],
    ...     dims=['warehouse', 'customer'],
    ...     coords={'warehouse': ['W1', 'W2'], 'customer': ['C1', 'C2']}
    ... )
    >>> cost_param = parameter_from_dataarray(cost, name='cost')
    >>> cost_param.get_value(('W1', 'C2'))
    15.0
    """
    _check_xarray()
    from cvxpy_or.sets import Parameter as ParameterClass

    if index is None:
        index = set_from_dataarray(da, name=name)

    # Flatten values in C-order (row-major) to match element ordering
    values = da.values.flatten(order="C")

    param_name = name or (str(da.name) if da.name is not None else None)
    param = ParameterClass(index, name=param_name)
    param.value = values
    return param


def variable_like_dataarray(
    da: xr.DataArray,
    name: str | None = None,
    nonneg: bool = False,
    **kwargs,
) -> Variable:
    """Create a Variable with the same shape/coords as a DataArray.

    Parameters
    ----------
    da : xr.DataArray
        Template DataArray defining the variable shape.
    name : str, optional
        Name for the Variable.
    nonneg : bool, optional
        If True, constrain variable to be non-negative.
    **kwargs
        Additional arguments passed to Variable.

    Returns
    -------
    Variable
        A new Variable indexed by the DataArray coordinates.

    Examples
    --------
    >>> import xarray as xr
    >>> cost = xr.DataArray(
    ...     [[10, 15], [12, 18]],
    ...     dims=['warehouse', 'customer'],
    ...     coords={'warehouse': ['W1', 'W2'], 'customer': ['C1', 'C2']}
    ... )
    >>> ship = variable_like_dataarray(cost, name='ship', nonneg=True)
    >>> ship.shape
    (4,)
    """
    _check_xarray()
    from cvxpy_or.sets import Variable as VariableClass

    index = set_from_dataarray(da, name=name)
    return VariableClass(index, nonneg=nonneg, name=name, **kwargs)


def variable_to_dataarray(
    var: Variable,
    name: str | None = None,
) -> xr.DataArray:
    """Convert a solved Variable to a DataArray.

    Parameters
    ----------
    var : Variable
        The Variable to convert (must be solved).
    name : str, optional
        Name for the DataArray. Defaults to variable name.

    Returns
    -------
    xr.DataArray
        DataArray with the variable's solution values.

    Raises
    ------
    ValueError
        If the Variable has no solution.

    Examples
    --------
    >>> prob.solve()
    >>> ship_da = variable_to_dataarray(ship)
    >>> ship_da.sel(warehouse='W1', customer='C1')
    """
    _check_xarray()

    if var.value is None:
        raise ValueError(f"Variable '{var.name()}' has no solution. Solve the problem first.")

    return _indexed_to_dataarray(var._set_index, var.value, name or var.name())


def parameter_to_dataarray(
    param: Parameter,
    name: str | None = None,
) -> xr.DataArray:
    """Convert a Parameter to a DataArray.

    Parameters
    ----------
    param : Parameter
        The Parameter to convert.
    name : str, optional
        Name for the DataArray.

    Returns
    -------
    xr.DataArray
        DataArray with the parameter's values.

    Raises
    ------
    ValueError
        If the Parameter has no data.
    """
    _check_xarray()

    if param.value is None:
        raise ValueError(f"Parameter '{param.name()}' has no data.")

    return _indexed_to_dataarray(param._set_index, param.value, name or param.name())


def _indexed_to_dataarray(
    index: Set,
    values: np.ndarray,
    name: str | None,
) -> xr.DataArray:
    """Convert Set-indexed values to a DataArray."""
    xr = _check_xarray()

    if not index._is_compound:
        # Single dimension
        dim_name = index.name or "index"
        coords = {dim_name: list(index._elements)}
        return xr.DataArray(values, dims=[dim_name], coords=coords, name=name)

    # Multi-dimensional - need to reshape
    if index._names is None:
        raise ValueError(
            "Compound index must have named dimensions for xarray export. "
            "Use Set(..., names=('dim1', 'dim2')) when creating the Set."
        )

    # Extract unique values per dimension (preserving order)
    dim_coords: dict[str, list[Any]] = {}
    for i, dim_name in enumerate(index._names):
        seen: set[Any] = set()
        dim_values: list[Any] = []
        for elem in index._elements:
            elem_tuple = cast(tuple[Any, ...], elem)
            val = elem_tuple[i]
            if val not in seen:
                dim_values.append(val)
                seen.add(val)
        dim_coords[dim_name] = dim_values

    # Compute shape
    shape = tuple(len(dim_coords[dim]) for dim in index._names)

    # Reshape values
    reshaped = np.asarray(values).reshape(shape)

    return xr.DataArray(
        reshaped,
        dims=list(index._names),
        coords=dim_coords,
        name=name,
    )
