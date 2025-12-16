"""Set-based indexing for CVXPY.

This module provides set-based indexing abstractions that feel natural to users
coming from AMPL/Pyomo. Variable and Parameter inherit from their CVXPY
counterparts, so all CVXPY operations work natively.

Example
-------
>>> from cvxpy_or import Set, Variable, Parameter
>>> import cvxpy as cp
>>>
>>> # Define index sets
>>> warehouses = Set(['W1', 'W2', 'W3'], name='warehouses')
>>> routes = Set.cross(warehouses, customers, name='routes')
>>>
>>> # Create indexed objects (they ARE cp.Variable/cp.Parameter)
>>> cost = Parameter(routes, data={('W1', 'C1'): 10, ('W1', 'C2'): 15, ...})
>>> ship = Variable(routes, nonneg=True)
>>>
>>> # All CVXPY operations work natively!
>>> objective = cost @ ship           # inner product
>>> constraint = ship >= 0            # variable constraint
>>> penalty = cp.abs(ship - target)   # atoms work
>>>
>>> # Named aggregation
>>> supply_constr = sum_by(ship, 'origin', index=routes) <= supply
"""

from __future__ import annotations

from itertools import product as itertools_product
from typing import Callable, Hashable, Iterable, Sequence

import numpy as np
import scipy.sparse as sp

import cvxpy as cp


def where(
    expr: cp.Expression,
    cond: np.ndarray | Callable[[Hashable], bool] | None = None,
    index: "Set | None" = None,
    **kwargs,
) -> cp.Expression:
    """Filter expression elements by condition, zeroing out non-matching entries.

    Parameters
    ----------
    expr : cp.Expression
        The expression to filter (Variable, Parameter, or any CVXPY expression).
    cond : np.ndarray or callable, optional
        Boolean array or callable that takes an index element and returns bool.
    index : Set, optional
        The Set indexing this expression. Required for callable or kwargs.
    **kwargs
        For compound indices, filter by position values.

    Returns
    -------
    cp.Expression
        An expression with non-matching elements zeroed out.

    Examples
    --------
    >>> where(ship, np.array([True, False, True]))
    >>> where(ship, lambda r: r[0] == 'W1', index=routes)
    >>> where(ship, index=routes, origin='W1')
    >>> where(2 * ship + cost, mask)  # Works on expressions
    """
    mask = _build_where_mask(cond, index, kwargs)
    return cp.multiply(mask, expr)


def sum_by(
    expr: cp.Expression,
    positions: int | str | list[int] | list[str],
    index: "Set",
) -> cp.Expression:
    """Aggregate expression by grouping on positions in compound index.

    For a compound index (tuples), this groups elements by the values
    at the specified positions and sums within each group.

    Parameters
    ----------
    expr : cp.Expression
        The expression to aggregate.
    positions : int, str, or list
        The position(s) to group by (dimensions to KEEP).
        Can be integers (0-indexed), string names, or a list of either.
    index : Set
        The Set indexing this expression. Must be compound (tuples).

    Returns
    -------
    cp.Expression
        A CVXPY expression with shape (n_groups,).

    Examples
    --------
    >>> sum_by(ship, 'origin', index=routes)  # Sum over destinations
    >>> sum_by(ship, ['origin', 'period'], index=idx)  # Keep multiple
    >>> sum_by(2 * ship, 0, index=routes)  # Works on expressions
    """
    if not index._is_compound:
        raise ValueError(
            f"sum_by() requires a compound index (tuples). "
            f"Set '{index.name}' contains simple elements. "
            f"Use cp.sum() to sum all elements."
        )

    # Normalize to list of position indices
    if isinstance(positions, (int, str)):
        positions = [positions]
    pos_indices = [index._resolve_position(p) for p in positions]

    # Build aggregation matrix
    agg_matrix = _build_aggregation_matrix(index, pos_indices)

    return agg_matrix @ expr


class Set:
    """An ordered set of elements for indexing variables and parameters.

    Parameters
    ----------
    elements : Iterable[Hashable]
        The elements of the index set. Can be simple values (strings, ints)
        or tuples for compound indices.
    name : str, optional
        A name for this index set (used in error messages).
    names : tuple[str, ...], optional
        Names for positions in compound (tuple) indices. For example,
        ``names=('origin', 'destination')`` allows ``sum_by('origin')``
        instead of ``sum_by(0)``.

    Examples
    --------
    Simple index:

    >>> warehouses = Set(['W1', 'W2', 'W3'], name='warehouses')
    >>> len(warehouses)
    3

    Compound index with named positions:

    >>> routes = Set(
    ...     [('W1', 'C1'), ('W1', 'C2'), ('W2', 'C1')],
    ...     name='routes',
    ...     names=('origin', 'destination')
    ... )
    >>> routes.position(('W1', 'C2'))
    1
    """

    def __init__(
        self,
        elements: Iterable[Hashable],
        name: str | None = None,
        names: Sequence[str] | None = None,
    ):
        self._elements = list(elements)
        self._name = name or f"Set_{id(self)}"
        self._pos = {e: i for i, e in enumerate(self._elements)}
        self._is_compound = (
            len(self._elements) > 0 and isinstance(self._elements[0], tuple)
        )
        self._names = tuple(names) if names else None

        # Validate names match arity of compound index
        if self._names and self._is_compound:
            arity = len(self._elements[0])
            if len(self._names) != arity:
                raise ValueError(
                    f"names has {len(self._names)} elements but index tuples "
                    f"have {arity} positions"
                )

    def __len__(self) -> int:
        return len(self._elements)

    def __iter__(self):
        return iter(self._elements)

    def __contains__(self, elem: Hashable) -> bool:
        return elem in self._pos

    def __repr__(self) -> str:
        if len(self._elements) <= 5:
            elems = str(self._elements)
        else:
            elems = f"[{self._elements[0]!r}, ..., {self._elements[-1]!r}] ({len(self)} elements)"
        return f"Set({elems}, name={self._name!r})"

    @property
    def name(self) -> str:
        return self._name

    @property
    def names(self) -> tuple[str, ...] | None:
        """Position names for compound indices."""
        return self._names

    def position(self, elem: Hashable) -> int:
        """Return the integer position of an element.

        Raises
        ------
        KeyError
            If the element is not in the index.
        """
        if elem not in self._pos:
            raise KeyError(f"Element {elem!r} not in index '{self._name}'")
        return self._pos[elem]

    def _resolve_position(self, key: int | str) -> int:
        """Convert a string name or int to a position index."""
        if isinstance(key, int):
            return key
        if self._names and key in self._names:
            return self._names.index(key)
        raise KeyError(
            f"Unknown position name: {key!r}. "
            f"Available names: {self._names}"
        )

    @staticmethod
    def cross(
        *indices: Set,
        name: str | None = None,
        names: Sequence[str] | None = None,
    ) -> Set:
        """Create cross-product of multiple indices.

        Parameters
        ----------
        *indices : Set
            Two or more Set objects to combine.
        name : str, optional
            Name for the resulting index.
        names : Sequence[str], optional
            Names for positions in the result. If not provided, uses the
            names of the source indices.

        Returns
        -------
        Set
            A new Set containing all combinations as tuples.

        Examples
        --------
        >>> warehouses = Set(['W1', 'W2'], name='warehouses')
        >>> customers = Set(['C1', 'C2', 'C3'], name='customers')
        >>> routes = Set.cross(warehouses, customers)
        >>> len(routes)
        6
        >>> list(routes)[:2]
        [('W1', 'C1'), ('W1', 'C2')]
        """
        if len(indices) < 2:
            raise ValueError("cross() requires at least 2 indices")

        elements = list(itertools_product(*[idx._elements for idx in indices]))

        # Auto-generate names from source index names if not provided
        if names is None:
            source_names = [idx._name for idx in indices]
            if all(n is not None for n in source_names):
                names = tuple(source_names)

        return Set(elements, name=name, names=names)


class Variable(cp.Variable):
    """A CVXPY Variable with Set-based indexing.

    This class inherits from cp.Variable, so all CVXPY operations work
    natively (arithmetic, constraints, atoms, etc.).

    Parameters
    ----------
    index : Set
        The index set for this variable.
    nonneg : bool, optional
        If True, constrain the variable to be non-negative.
    name : str, optional
        Name for the variable.
    **kwargs
        Additional keyword arguments passed to ``cp.Variable``.

    Examples
    --------
    >>> routes = Set(
    ...     [('W1', 'C1'), ('W1', 'C2'), ('W2', 'C1')],
    ...     name='routes',
    ...     names=('origin', 'destination')
    ... )
    >>> ship = Variable(routes, nonneg=True, name='ship')
    >>>
    >>> # All CVXPY operations work!
    >>> ship >= 0                    # constraint
    >>> cp.sum(ship)                 # atom
    >>> cost @ ship                  # inner product
    >>>
    >>> # Named indexing
    >>> ship[('W1', 'C1')]           # named access
    """

    def __init__(
        self,
        index: Set,
        nonneg: bool = False,
        name: str | None = None,
        **kwargs,
    ):
        self._set_index = index
        super().__init__(len(index), nonneg=nonneg, name=name, **kwargs)

    @property
    def index(self) -> Set:
        """The Set indexing this variable."""
        return self._set_index

    def __getitem__(self, key):
        """Access element by index key or standard CVXPY indexing.

        If key is in the Set, returns the element at that position.
        Otherwise, delegates to standard CVXPY indexing (slices, etc.).
        """
        if key in self._set_index:
            return super().__getitem__(self._set_index.position(key))
        return super().__getitem__(key)

    def get_value(self, key: Hashable) -> float | None:
        """Get the solved value for a specific index element.

        Parameters
        ----------
        key : Hashable
            An element of the index Set.

        Returns
        -------
        float | None
            The value at that index, or None if not solved yet.
        """
        if self.value is None:
            return None
        return float(self.value[self._set_index.position(key)])

    def __repr__(self) -> str:
        return f"Variable(index={self._set_index.name!r}, shape={self.shape})"


class Parameter(cp.Parameter):
    """A CVXPY Parameter with Set-based indexing.

    This class inherits from cp.Parameter, so all CVXPY operations work
    natively (arithmetic, constraints, atoms, etc.).

    Parameters
    ----------
    index : Set
        The index set for this parameter.
    data : dict[Hashable, float], optional
        Initial values as a dict mapping index elements to values.
    name : str, optional
        Name for the parameter.
    **kwargs
        Additional keyword arguments passed to ``cp.Parameter``.

    Examples
    --------
    >>> routes = Set([('W1', 'C1'), ('W1', 'C2')], name='routes')
    >>> cost = Parameter(routes, data={('W1', 'C1'): 10, ('W1', 'C2'): 20})
    >>>
    >>> # All CVXPY operations work!
    >>> cost @ ship                  # inner product
    >>> cp.multiply(cost, ship)      # element-wise multiply
    >>> cost + 5                     # scalar arithmetic
    """

    def __init__(
        self,
        index: Set,
        data: dict[Hashable, float] | None = None,
        name: str | None = None,
        **kwargs,
    ):
        self._set_index = index
        super().__init__(len(index), name=name, **kwargs)
        if data is not None:
            self.set_data(data)

    @property
    def index(self) -> Set:
        """The Set indexing this parameter."""
        return self._set_index

    def set_data(self, data: dict[Hashable, float]) -> None:
        """Set parameter values from a dict.

        Parameters
        ----------
        data : dict[Hashable, float]
            A dict mapping index elements to values.
        """
        values = np.zeros(len(self._set_index))
        for elem, val in data.items():
            pos = self._set_index.position(elem)
            values[pos] = val
        self.value = values

    def __getitem__(self, key):
        """Access element by index key or standard CVXPY indexing.

        If key is in the Set, returns the element at that position.
        Otherwise, delegates to standard CVXPY indexing (slices, etc.).
        """
        if key in self._set_index:
            return super().__getitem__(self._set_index.position(key))
        return super().__getitem__(key)

    def get_value(self, key: Hashable) -> float | None:
        """Get the value for a specific index element.

        Parameters
        ----------
        key : Hashable
            An element of the index Set.

        Returns
        -------
        float | None
            The value at that index, or None if not set yet.
        """
        if self.value is None:
            return None
        return float(self.value[self._set_index.position(key)])

    def expand(self, target_index: Set, positions: list[int] | list[str]) -> Parameter:
        """Expand (broadcast) this parameter to a larger cross-product index.

        Creates a new parameter indexed by `target_index` where values are
        looked up from this parameter based on the specified positions.

        Parameters
        ----------
        target_index : Set
            The target cross-product index to expand to.
        positions : list[int] | list[str]
            Which positions in the target index correspond to this parameter's
            index. For a 1D parameter expanding to 2D, use a single-element list.

        Returns
        -------
        Parameter
            A new parameter with the expanded index.

        Examples
        --------
        Expand 1D holding cost to 2D (warehouse, period):

        >>> holding_cost = Parameter(warehouses, data={'W1': 0.1, 'W2': 0.2})
        >>> inv_idx = Set.cross(warehouses, periods)
        >>> holding_cost_2d = holding_cost.expand(inv_idx, [0])
        >>> # Now holding_cost_2d[('W1', 'Jan')] == 0.1

        Expand 2D route cost to 3D (warehouse, customer, period):

        >>> cost = Parameter(routes, data={...})
        >>> shipments = Set.cross(warehouses, customers, periods)
        >>> cost_3d = cost.expand(shipments, [0, 1])
        """
        if not target_index._is_compound:
            raise ValueError("Target index must be a compound (cross-product) index")

        # Normalize positions to integers
        pos_indices = [target_index._resolve_position(p) for p in positions]

        # Build lookup: for each element in target_index, find the key in self
        result_values = np.zeros(len(target_index))

        for i, elem in enumerate(target_index):
            # Extract the key for this parameter from the target element
            if len(pos_indices) == 1:
                key = elem[pos_indices[0]]
            else:
                key = tuple(elem[p] for p in pos_indices)

            # Look up value from this parameter
            src_pos = self._set_index.position(key)
            result_values[i] = self.value[src_pos]

        result = Parameter(target_index)
        result.value = result_values
        return result

    def __repr__(self) -> str:
        return f"Parameter(index={self._set_index.name!r}, shape={self.shape})"


def _build_aggregation_matrix(
    index: Set, pos_indices: list[int]
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
    # Function to extract group key from element
    def get_key(elem: tuple) -> Hashable:
        if len(pos_indices) == 1:
            return elem[pos_indices[0]]
        return tuple(elem[i] for i in pos_indices)

    # Find unique keys (preserving order of first occurrence)
    group_keys: list[Hashable] = []
    seen: set[Hashable] = set()
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


def _build_where_mask(
    cond: np.ndarray | Callable[[Hashable], bool] | None,
    index: Set | None,
    kwargs: dict,
) -> np.ndarray:
    """Build boolean mask array for where() filtering.

    Parameters
    ----------
    cond : np.ndarray, callable, or None
        Direct condition (array or callable).
    index : Set or None
        The index set. Required for callable or kwargs.
    kwargs : dict
        Position-based filtering for compound indices.

    Returns
    -------
    np.ndarray
        Float array of 1.0 (included) and 0.0 (excluded).
    """
    if cond is not None and kwargs:
        raise ValueError("Cannot specify both cond and keyword arguments")

    if cond is not None:
        if callable(cond):
            if index is None:
                raise ValueError(
                    "index parameter is required when cond is a callable"
                )
            mask = np.array([cond(elem) for elem in index], dtype=float)
        else:
            mask = np.asarray(cond, dtype=float)
            if index is not None and mask.shape != (len(index),):
                raise ValueError(
                    f"Condition array has shape {mask.shape}, "
                    f"expected ({len(index)},)"
                )
    elif kwargs:
        if index is None:
            raise ValueError(
                "index parameter is required when using keyword filtering"
            )
        if not index._is_compound:
            raise ValueError(
                "Keyword filtering requires a compound index (tuples). "
                f"Set '{index.name}' contains simple elements."
            )

        n = len(index)
        mask = np.ones(n, dtype=float)
        for key, allowed in kwargs.items():
            pos = index._resolve_position(key)
            if not isinstance(allowed, (list, tuple, set)):
                allowed = {allowed}
            else:
                allowed = set(allowed)
            for i, elem in enumerate(index):
                if elem[pos] not in allowed:
                    mask[i] = 0.0
    else:
        raise ValueError("Must specify either cond or keyword arguments")

    return mask
