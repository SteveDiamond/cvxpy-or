"""I/O utilities for cvxpy-or.

This module provides functions for loading Sets and Parameters from
pandas DataFrames and exporting Variable values back to DataFrames.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import TYPE_CHECKING, Any, cast

import pandas as pd

if TYPE_CHECKING:
    from cvxpy_or.sets import Parameter, Set, Variable


def set_from_series(series: pd.Series, name: str | None = None) -> Set:
    """Create a Set from a pandas Series.

    Parameters
    ----------
    series : pd.Series
        The series whose unique values become Set elements.
    name : str, optional
        Name for the Set. Defaults to series name.

    Returns
    -------
    Set
        A new Set with unique values from the series.

    Examples
    --------
    >>> df = pd.DataFrame({'customer': ['C1', 'C2', 'C1', 'C3']})
    >>> customers = set_from_series(df['customer'])
    >>> list(customers)
    ['C1', 'C2', 'C3']
    """
    from cvxpy_or.sets import Set as SetClass

    elements = series.unique().tolist()
    set_name = name or (str(series.name) if series.name is not None else None)
    return SetClass(elements, name=set_name)


def set_from_dataframe(
    df: pd.DataFrame,
    columns: Sequence[str],
    name: str | None = None,
    names: Sequence[str] | None = None,
) -> Set:
    """Create a compound Set from DataFrame columns.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to extract tuples from.
    columns : sequence of str
        Column names to combine into tuples.
    name : str, optional
        Name for the Set.
    names : sequence of str, optional
        Names for positions. Defaults to column names.

    Returns
    -------
    Set
        A new compound Set.

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'origin': ['W1', 'W1', 'W2'],
    ...     'dest': ['C1', 'C2', 'C1'],
    ...     'cost': [10, 15, 20]
    ... })
    >>> routes = set_from_dataframe(df, ['origin', 'dest'])
    >>> list(routes)
    [('W1', 'C1'), ('W1', 'C2'), ('W2', 'C1')]
    """
    from cvxpy_or.sets import Set as SetClass

    # Extract unique tuples
    tuples = df[list(columns)].drop_duplicates().itertuples(index=False, name=None)
    elements = list(tuples)

    # Use column names as position names if not provided
    if names is None:
        names = tuple(columns)

    return SetClass(elements, name=name, names=names)


def set_from_index(
    df: pd.DataFrame,
    name: str | None = None,
    names: Sequence[str] | None = None,
) -> Set:
    """Create a Set from a DataFrame's index.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame whose index becomes Set elements.
    name : str, optional
        Name for the Set.
    names : sequence of str, optional
        Names for positions (for MultiIndex).

    Returns
    -------
    Set
        A new Set from the index.
    """
    from cvxpy_or.sets import Set as SetClass

    if isinstance(df.index, pd.MultiIndex):
        elements = list(df.index)
        if names is None:
            # Convert Hashable names to strings
            names = tuple(
                str(n) if n is not None else f"level_{i}" for i, n in enumerate(df.index.names)
            )
    else:
        elements = list(df.index)

    set_name = name or (str(df.index.name) if df.index.name is not None else None)
    return SetClass(elements, name=set_name, names=names)


def parameter_from_dataframe(
    df: pd.DataFrame,
    index_cols: str | Sequence[str],
    value_col: str,
    index: Set | None = None,
    name: str | None = None,
) -> Parameter:
    """Create a Parameter from a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the data.
    index_cols : str or sequence of str
        Column(s) to use as the index.
    value_col : str
        Column containing the parameter values.
    index : Set, optional
        Existing Set to use. If not provided, creates one from data.
    name : str, optional
        Name for the Parameter.

    Returns
    -------
    Parameter
        A new Parameter with data from the DataFrame.

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'origin': ['W1', 'W1', 'W2'],
    ...     'dest': ['C1', 'C2', 'C1'],
    ...     'cost': [10, 15, 20]
    ... })
    >>> cost = parameter_from_dataframe(df, ['origin', 'dest'], 'cost')
    """
    from cvxpy_or.sets import Parameter as ParameterClass

    # Normalize index_cols to list
    if isinstance(index_cols, str):
        index_cols = [index_cols]

    # Create index if not provided
    if index is None:
        index = set_from_dataframe(df, index_cols, name=name)

    # Build data dict
    data: dict[Hashable, float] = {}
    for _, row in df.iterrows():
        if len(index_cols) == 1:
            key: Hashable = cast(Hashable, row[index_cols[0]])
        else:
            key = tuple(row[col] for col in index_cols)
        data[key] = float(row[value_col])

    return ParameterClass(index, data=data, name=name)


def parameter_from_series(
    series: pd.Series,
    index: Set | None = None,
    name: str | None = None,
) -> Parameter:
    """Create a Parameter from a pandas Series.

    Parameters
    ----------
    series : pd.Series
        The Series with index as keys and values as parameter data.
    index : Set, optional
        Existing Set to use. If not provided, creates from series index.
    name : str, optional
        Name for the Parameter.

    Returns
    -------
    Parameter
        A new Parameter with data from the Series.

    Examples
    --------
    >>> supply = pd.Series({'W1': 100, 'W2': 150, 'W3': 200})
    >>> param = parameter_from_series(supply, name='supply')
    """
    from cvxpy_or.sets import Parameter as ParameterClass
    from cvxpy_or.sets import Set as SetClass

    # Create index if not provided
    if index is None:
        if isinstance(series.index, pd.MultiIndex):
            elements = list(series.index)
            idx_names: tuple[str, ...] | None = tuple(
                str(n) if n is not None else f"level_{i}" for i, n in enumerate(series.index.names)
            )
        else:
            elements = list(series.index)
            idx_names = None
        set_name = str(series.name) if series.name is not None else name
        index = SetClass(elements, name=set_name, names=idx_names)

    # Build data dict - convert series to dict with proper types
    data: dict[Hashable, float] = {k: float(v) for k, v in series.items()}

    param_name = name or (str(series.name) if series.name is not None else None)
    return ParameterClass(index, data=data, name=param_name)


def variable_to_dataframe(
    var: Variable,
    value_col: str = "value",
) -> pd.DataFrame:
    """Convert a Variable's solution values to a DataFrame.

    Parameters
    ----------
    var : Variable
        The Variable to convert (must be solved).
    value_col : str, optional
        Name for the value column. Default "value".

    Returns
    -------
    pd.DataFrame
        DataFrame with index columns and values.

    Raises
    ------
    ValueError
        If the Variable has no solution.

    Examples
    --------
    >>> prob.solve()
    >>> df = variable_to_dataframe(ship)
    >>> df.head()
       warehouse  customer    value
    0        W1        C1    50.0
    1        W1        C2    30.0
    """
    if var.value is None:
        raise ValueError(f"Variable '{var.name}' has no solution. Solve the problem first.")

    index = var._set_index

    # Build rows
    rows = []
    for elem in index:
        pos = index.position(elem)
        value = float(var.value[pos])
        if isinstance(elem, tuple):
            row = list(elem) + [value]
        else:
            row = [elem, value]
        rows.append(row)

    # Build column names
    if index._is_compound and index._names:
        columns = list(index._names) + [value_col]
    elif index._is_compound:
        first_elem = cast(tuple[Any, ...], index._elements[0])
        columns = [f"pos_{i}" for i in range(len(first_elem))] + [value_col]
    else:
        columns = [index.name or "index", value_col]

    return pd.DataFrame(rows, columns=columns)  # type: ignore[arg-type]


def parameter_to_dataframe(
    param: Parameter,
    value_col: str = "value",
) -> pd.DataFrame:
    """Convert a Parameter's values to a DataFrame.

    Parameters
    ----------
    param : Parameter
        The Parameter to convert.
    value_col : str, optional
        Name for the value column.

    Returns
    -------
    pd.DataFrame
        DataFrame with index columns and values.
    """
    if param.value is None:
        raise ValueError(
            f"Parameter '{param.name}' has no data. Set data with param.set_data({{...}})."
        )

    index = param._set_index

    # Build rows
    rows = []
    for elem in index:
        pos = index.position(elem)
        value = float(param.value[pos])
        if isinstance(elem, tuple):
            row = list(elem) + [value]
        else:
            row = [elem, value]
        rows.append(row)

    # Build column names
    if index._is_compound and index._names:
        columns = list(index._names) + [value_col]
    elif index._is_compound:
        first_elem = cast(tuple[Any, ...], index._elements[0])
        columns = [f"pos_{i}" for i in range(len(first_elem))] + [value_col]
    else:
        columns = [index.name or "index", value_col]

    return pd.DataFrame(rows, columns=columns)  # type: ignore[arg-type]
