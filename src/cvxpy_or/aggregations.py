"""Aggregation functions for cvxpy-or.

This module provides group-by aggregation functions that extend sum_by
with additional operations like mean, count, any, and all.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse as sp

import cvxpy as cp

if TYPE_CHECKING:
    from cvxpy_or.sets import Set


def _infer_index(expr: cp.Expression) -> "Set":
    """Infer the Set index from Variables/Parameters in an expression tree.

    Import from sets module to avoid circular imports.
    """
    from cvxpy_or.sets import Parameter, Variable

    indices: set[Set] = set()

    def walk(node):
        if isinstance(node, (Variable, Parameter)):
            indices.add(node._set_index)
        if hasattr(node, "args"):
            for arg in node.args:
                walk(arg)

    walk(expr)

    if len(indices) == 0:
        raise TypeError(
            "Cannot infer index: expression contains no Variable or Parameter."
        )
    if len(indices) > 1:
        names = [idx.name for idx in indices]
        raise TypeError(
            f"Cannot infer index: expression contains objects from "
            f"different indices ({names})."
        )
    return indices.pop()


def _get_group_info(index: "Set", positions: int | str | list[int] | list[str]):
    """Get group keys and sizes for aggregation.

    Returns
    -------
    tuple
        (group_keys, key_to_row, group_sizes)
    """
    # Normalize positions
    if isinstance(positions, (int, str)):
        positions = [positions]
    pos_indices = [index._resolve_position(p) for p in positions]

    def get_key(elem: tuple):
        if len(pos_indices) == 1:
            return elem[pos_indices[0]]
        return tuple(elem[i] for i in pos_indices)

    # Find unique keys and count group sizes
    group_keys: list = []
    key_to_row: dict = {}
    group_sizes: dict = {}

    for elem in index:
        key = get_key(elem)
        if key not in key_to_row:
            key_to_row[key] = len(group_keys)
            group_keys.append(key)
            group_sizes[key] = 0
        group_sizes[key] += 1

    return group_keys, key_to_row, group_sizes, pos_indices


def _build_aggregation_matrix(
    index: "Set", pos_indices: list[int]
) -> sp.csr_matrix:
    """Build a sparse aggregation matrix for sum_by.

    Parameters
    ----------
    index : Set
        The source index (must be compound).
    pos_indices : list[int]
        Positions to group by.

    Returns
    -------
    sp.csr_matrix
        Aggregation matrix of shape (n_groups, len(index)).
    """

    def get_key(elem: tuple):
        if len(pos_indices) == 1:
            return elem[pos_indices[0]]
        return tuple(elem[i] for i in pos_indices)

    # Find unique keys (preserving order of first occurrence)
    group_keys: list = []
    seen: set = set()
    for elem in index:
        key = get_key(elem)
        if key not in seen:
            group_keys.append(key)
            seen.add(key)

    # Build sparse matrix
    n_groups = len(group_keys)
    n_elements = len(index)

    key_to_row = {k: i for i, k in enumerate(group_keys)}

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    for j, elem in enumerate(index):
        key = get_key(elem)
        row = key_to_row[key]
        rows.append(row)
        cols.append(j)
        data.append(1.0)

    return sp.csr_matrix((data, (rows, cols)), shape=(n_groups, n_elements))


def mean_by(
    expr: cp.Expression,
    positions: int | str | list[int] | list[str],
) -> cp.Expression:
    """Compute mean by grouping on positions in compound index.

    For a compound index (tuples), this groups elements by the values
    at the specified positions and computes the mean within each group.

    Parameters
    ----------
    expr : cp.Expression
        The expression to aggregate.
    positions : int, str, or list
        The position(s) to group by (dimensions to KEEP).

    Returns
    -------
    cp.Expression
        A CVXPY expression with shape (n_groups,).

    Examples
    --------
    >>> mean_by(cost, 'warehouse')  # Average cost per warehouse
    >>> mean_by(ship, ['origin', 'period'])  # Mean over destinations
    """
    index = _infer_index(expr)

    if not index._is_compound:
        raise ValueError(
            f"mean_by() requires a compound index (tuples). "
            f"Set '{index.name}' contains simple elements. "
            f"Use cp.mean() to compute mean of all elements."
        )

    group_keys, key_to_row, group_sizes, pos_indices = _get_group_info(
        index, positions
    )

    # Build mean matrix (like sum, but divided by group size)
    n_groups = len(group_keys)
    n_elements = len(index)

    def get_key(elem: tuple):
        if len(pos_indices) == 1:
            return elem[pos_indices[0]]
        return tuple(elem[i] for i in pos_indices)

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    for j, elem in enumerate(index):
        key = get_key(elem)
        row = key_to_row[key]
        rows.append(row)
        cols.append(j)
        data.append(1.0 / group_sizes[key])  # Divide by group size

    agg_matrix = sp.csr_matrix((data, (rows, cols)), shape=(n_groups, n_elements))
    return agg_matrix @ expr


def count_by(
    index: "Set",
    positions: int | str | list[int] | list[str],
) -> np.ndarray:
    """Count elements per group in a compound index.

    Unlike other aggregation functions, this operates on the Set itself
    (not an expression) and returns a numpy array.

    Parameters
    ----------
    index : Set
        The index to count elements in.
    positions : int, str, or list
        The position(s) to group by.

    Returns
    -------
    np.ndarray
        Array of counts per group.

    Examples
    --------
    >>> routes = Set.cross(warehouses, customers)
    >>> count_by(routes, 'warehouse')  # Customers per warehouse
    array([3, 3, 3])  # 3 customers per warehouse
    """
    if not index._is_compound:
        raise ValueError(
            f"count_by() requires a compound index (tuples). "
            f"Set '{index.name}' contains simple elements."
        )

    group_keys, key_to_row, group_sizes, _ = _get_group_info(index, positions)
    return np.array([group_sizes[k] for k in group_keys])


def group_keys(
    index: "Set",
    positions: int | str | list[int] | list[str],
) -> list:
    """Get the unique group keys for aggregation.

    Parameters
    ----------
    index : Set
        The compound index.
    positions : int, str, or list
        The position(s) to group by.

    Returns
    -------
    list
        List of unique keys (values or tuples).

    Examples
    --------
    >>> routes = Set.cross(warehouses, customers)
    >>> group_keys(routes, 'warehouse')
    ['W1', 'W2', 'W3']
    """
    if not index._is_compound:
        raise ValueError(
            f"group_keys() requires a compound index (tuples). "
            f"Set '{index.name}' contains simple elements."
        )

    group_keys_list, _, _, _ = _get_group_info(index, positions)
    return group_keys_list


def max_by(
    expr: cp.Expression,
    positions: int | str | list[int] | list[str],
    *,
    aux_var_name: str | None = None,
) -> tuple[cp.Variable, list[cp.Constraint]]:
    """Compute maximum by grouping, returning auxiliary variable and constraints.

    Because CVXPY doesn't have a native group-max, this returns:
    1. An auxiliary variable representing the max per group
    2. Constraints that enforce the max relationship

    Add the returned constraints to your problem.

    Parameters
    ----------
    expr : cp.Expression
        The expression to take max of.
    positions : int, str, or list
        The position(s) to group by.
    aux_var_name : str, optional
        Name for the auxiliary variable.

    Returns
    -------
    tuple[cp.Variable, list[cp.Constraint]]
        (max_var, constraints) where max_var[i] is the max of group i.

    Examples
    --------
    >>> max_ship, constraints = max_by(ship, 'warehouse')
    >>> prob = cp.Problem(
    ...     cp.Minimize(cp.sum(max_ship)),  # Minimize max shipment per warehouse
    ...     constraints + other_constraints
    ... )
    """
    index = _infer_index(expr)

    if not index._is_compound:
        raise ValueError(
            f"max_by() requires a compound index (tuples). "
            f"Set '{index.name}' contains simple elements."
        )

    group_keys_list, key_to_row, _, pos_indices = _get_group_info(index, positions)
    n_groups = len(group_keys_list)

    def get_key(elem: tuple):
        if len(pos_indices) == 1:
            return elem[pos_indices[0]]
        return tuple(elem[i] for i in pos_indices)

    # Create auxiliary variable for max per group
    max_var = cp.Variable(n_groups, name=aux_var_name)

    # Constraints: max_var[g] >= expr[i] for all i in group g
    constraints = []
    for j, elem in enumerate(index):
        key = get_key(elem)
        row = key_to_row[key]
        constraints.append(max_var[row] >= expr[j])

    return max_var, constraints


def min_by(
    expr: cp.Expression,
    positions: int | str | list[int] | list[str],
    *,
    aux_var_name: str | None = None,
) -> tuple[cp.Variable, list[cp.Constraint]]:
    """Compute minimum by grouping, returning auxiliary variable and constraints.

    Because CVXPY doesn't have a native group-min, this returns:
    1. An auxiliary variable representing the min per group
    2. Constraints that enforce the min relationship

    Add the returned constraints to your problem.

    Parameters
    ----------
    expr : cp.Expression
        The expression to take min of.
    positions : int, str, or list
        The position(s) to group by.
    aux_var_name : str, optional
        Name for the auxiliary variable.

    Returns
    -------
    tuple[cp.Variable, list[cp.Constraint]]
        (min_var, constraints) where min_var[i] is the min of group i.

    Examples
    --------
    >>> min_ship, constraints = min_by(ship, 'warehouse')
    >>> prob = cp.Problem(
    ...     cp.Maximize(cp.sum(min_ship)),  # Maximize min shipment per warehouse
    ...     constraints + other_constraints
    ... )
    """
    index = _infer_index(expr)

    if not index._is_compound:
        raise ValueError(
            f"min_by() requires a compound index (tuples). "
            f"Set '{index.name}' contains simple elements."
        )

    group_keys_list, key_to_row, _, pos_indices = _get_group_info(index, positions)
    n_groups = len(group_keys_list)

    def get_key(elem: tuple):
        if len(pos_indices) == 1:
            return elem[pos_indices[0]]
        return tuple(elem[i] for i in pos_indices)

    # Create auxiliary variable for min per group
    min_var = cp.Variable(n_groups, name=aux_var_name)

    # Constraints: min_var[g] <= expr[i] for all i in group g
    constraints = []
    for j, elem in enumerate(index):
        key = get_key(elem)
        row = key_to_row[key]
        constraints.append(min_var[row] <= expr[j])

    return min_var, constraints


def sum_by_expr(
    expr: cp.Expression,
    positions: int | str | list[int] | list[str],
) -> cp.Expression:
    """Aggregate expression by grouping on positions in compound index.

    This is an alias for the main sum_by in sets.py, provided here
    for completeness of the aggregations module.

    Parameters
    ----------
    expr : cp.Expression
        The expression to aggregate.
    positions : int, str, or list
        The position(s) to group by.

    Returns
    -------
    cp.Expression
        A CVXPY expression with shape (n_groups,).
    """
    index = _infer_index(expr)

    if not index._is_compound:
        raise ValueError(
            f"sum_by_expr() requires a compound index (tuples). "
            f"Set '{index.name}' contains simple elements. "
            f"Use cp.sum() to sum all elements."
        )

    # Normalize positions
    if isinstance(positions, (int, str)):
        positions = [positions]
    pos_indices = [index._resolve_position(p) for p in positions]

    agg_matrix = _build_aggregation_matrix(index, pos_indices)
    return agg_matrix @ expr
