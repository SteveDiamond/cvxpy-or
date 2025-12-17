"""Display utilities for cvxpy-or.

This module provides pretty printing for Variables, Parameters, and
optimization results.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from cvxpy_or.sets import Parameter, Set, Variable

# Try to import rich for fancy tables, fall back to simple ASCII
try:
    from rich.console import Console
    from rich.table import Table

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


def _get_name(obj) -> str:
    """Get the name of a CVXPY object safely."""
    # Try _name first (our custom attribute)
    if hasattr(obj, '_name') and obj._name:
        return obj._name
    # Try name() as a method
    name_attr = getattr(obj, 'name', None)
    if callable(name_attr):
        result = name_attr()
        if result:
            return str(result)
    elif name_attr:
        return str(name_attr)
    return 'unnamed'


def format_value(value: float | None, precision: int = 4) -> str:
    """Format a numeric value for display.

    Parameters
    ----------
    value : float or None
        The value to format.
    precision : int
        Number of decimal places.

    Returns
    -------
    str
        Formatted value string.
    """
    if value is None:
        return "None"
    if abs(value) < 1e-10:
        return "0"
    if abs(value) > 1e6 or (abs(value) < 1e-3 and value != 0):
        return f"{value:.{precision}e}"
    return f"{value:.{precision}f}"


def variable_table(
    var: Variable,
    *,
    title: str | None = None,
    show_zero: bool = True,
    precision: int = 4,
    filter_fn: Callable[[Any], bool] | None = None,
) -> str:
    """Create a table representation of a Variable's values.

    Parameters
    ----------
    var : Variable
        The Variable to display.
    title : str, optional
        Title for the table. Defaults to variable name.
    show_zero : bool, optional
        Whether to show rows with zero values. Default True.
    precision : int, optional
        Decimal precision for values. Default 4.
    filter_fn : callable, optional
        Function to filter which elements to show.

    Returns
    -------
    str
        Table as a string (rich table if available, ASCII otherwise).
    """
    if var.value is None:
        var_name = _get_name(var)
        return f"Variable '{var_name}' has no solution yet (solve the problem first)"

    index = var._set_index
    var_name = _get_name(var)
    title = title or f"Variable: {var_name}"

    rows = []
    for elem in index:
        if filter_fn is not None and not filter_fn(elem):
            continue
        pos = index.position(elem)
        value = float(var.value[pos])
        if not show_zero and abs(value) < 1e-6:
            continue
        rows.append((elem, value))

    if not rows:
        return f"{title}\n(no values to display)"

    return _format_table(title, index, rows, precision)


def parameter_table(
    param: Parameter,
    *,
    title: str | None = None,
    precision: int = 4,
    filter_fn: Callable[[Any], bool] | None = None,
) -> str:
    """Create a table representation of a Parameter's values.

    Parameters
    ----------
    param : Parameter
        The Parameter to display.
    title : str, optional
        Title for the table.
    precision : int, optional
        Decimal precision for values.
    filter_fn : callable, optional
        Function to filter which elements to show.

    Returns
    -------
    str
        Table as a string.
    """
    if param.value is None:
        param_name = _get_name(param)
        return f"Parameter '{param_name}' has no data set"

    index = param._set_index
    param_name = _get_name(param)
    title = title or f"Parameter: {param_name}"

    rows = []
    for elem in index:
        if filter_fn is not None and not filter_fn(elem):
            continue
        pos = index.position(elem)
        value = float(param.value[pos])
        rows.append((elem, value))

    if not rows:
        return f"{title}\n(no values to display)"

    return _format_table(title, index, rows, precision)


def _format_table(
    title: str,
    index: Set,
    rows: list[tuple[Any, float]],
    precision: int,
) -> str:
    """Format rows as a table.

    Uses rich if available, otherwise ASCII.
    """
    if RICH_AVAILABLE:
        return _format_rich_table(title, index, rows, precision)
    return _format_ascii_table(title, index, rows, precision)


def _format_rich_table(
    title: str,
    index: Set,
    rows: list[tuple[Any, float]],
    precision: int,
) -> str:
    """Format table using rich library."""
    console = Console(force_terminal=False, width=120)
    table = Table(title=title, show_header=True, header_style="bold")

    # Add columns based on index structure
    if index._is_compound and index._names:
        for name in index._names:
            table.add_column(name)
    elif index._is_compound:
        first_elem = rows[0][0]
        for i in range(len(first_elem)):
            table.add_column(f"pos_{i}")
    else:
        table.add_column("index")

    table.add_column("value", justify="right")

    # Add rows
    for elem, value in rows:
        if isinstance(elem, tuple):
            row_values = [str(e) for e in elem] + [format_value(value, precision)]
        else:
            row_values = [str(elem), format_value(value, precision)]
        table.add_row(*row_values)

    # Render to string
    with console.capture() as capture:
        console.print(table)
    return capture.get()


def _format_ascii_table(
    title: str,
    index: Set,
    rows: list[tuple[Any, float]],
    precision: int,
) -> str:
    """Format table using ASCII characters."""
    lines = [title, "=" * len(title)]

    # Determine columns
    if index._is_compound and index._names:
        headers = list(index._names) + ["value"]
    elif index._is_compound:
        first_elem = rows[0][0]
        headers = [f"pos_{i}" for i in range(len(first_elem))] + ["value"]
    else:
        headers = ["index", "value"]

    # Calculate column widths
    col_data: list[list[str]] = [[] for _ in headers]
    for elem, value in rows:
        if isinstance(elem, tuple):
            for i, e in enumerate(elem):
                col_data[i].append(str(e))
        else:
            col_data[0].append(str(elem))
        col_data[-1].append(format_value(value, precision))

    widths = [
        max(len(h), max((len(s) for s in col), default=0))
        for h, col in zip(headers, col_data)
    ]

    # Format header
    header_row = " | ".join(h.ljust(w) for h, w in zip(headers, widths))
    lines.append(header_row)
    lines.append("-+-".join("-" * w for w in widths))

    # Format data rows
    for _i, (elem, value) in enumerate(rows):
        if isinstance(elem, tuple):
            row_values = [str(e) for e in elem] + [format_value(value, precision)]
        else:
            row_values = [str(elem), format_value(value, precision)]
        row_str = " | ".join(v.ljust(w) for v, w in zip(row_values, widths))
        lines.append(row_str)

    return "\n".join(lines)


def print_variable(
    var: Variable,
    *,
    title: str | None = None,
    show_zero: bool = True,
    precision: int = 4,
    filter_fn: Callable[[Any], bool] | None = None,
) -> None:
    """Print a Variable's values as a table.

    Parameters
    ----------
    var : Variable
        The Variable to print.
    title : str, optional
        Title for the table.
    show_zero : bool, optional
        Whether to show rows with zero values.
    precision : int, optional
        Decimal precision for values.
    filter_fn : callable, optional
        Function to filter which elements to show.
    """
    print(variable_table(var, title=title, show_zero=show_zero,
                         precision=precision, filter_fn=filter_fn))


def print_parameter(
    param: Parameter,
    *,
    title: str | None = None,
    precision: int = 4,
    filter_fn: Callable[[Any], bool] | None = None,
) -> None:
    """Print a Parameter's values as a table.

    Parameters
    ----------
    param : Parameter
        The Parameter to print.
    title : str, optional
        Title for the table.
    precision : int, optional
        Decimal precision for values.
    filter_fn : callable, optional
        Function to filter which elements to show.
    """
    print(parameter_table(param, title=title, precision=precision,
                          filter_fn=filter_fn))


def solution_summary(
    variables: Sequence[Variable],
    *,
    objective_value: float | None = None,
    status: str | None = None,
    show_zero: bool = False,
    precision: int = 4,
) -> str:
    """Create a summary of optimization solution.

    Parameters
    ----------
    variables : sequence of Variable
        The variables to summarize.
    objective_value : float, optional
        The objective value.
    status : str, optional
        The solver status.
    show_zero : bool, optional
        Whether to show zero values.
    precision : int, optional
        Decimal precision.

    Returns
    -------
    str
        Summary as a string.
    """
    lines = ["=" * 50]
    lines.append("SOLUTION SUMMARY")
    lines.append("=" * 50)

    if status is not None:
        lines.append(f"Status: {status}")
    if objective_value is not None:
        lines.append(f"Objective: {format_value(objective_value, precision)}")

    lines.append("")

    for var in variables:
        lines.append(variable_table(var, show_zero=show_zero, precision=precision))
        lines.append("")

    return "\n".join(lines)


def print_solution(
    variables: Sequence[Variable],
    *,
    objective_value: float | None = None,
    status: str | None = None,
    show_zero: bool = False,
    precision: int = 4,
) -> None:
    """Print a summary of optimization solution.

    Parameters
    ----------
    variables : sequence of Variable
        The variables to summarize.
    objective_value : float, optional
        The objective value.
    status : str, optional
        The solver status.
    show_zero : bool, optional
        Whether to show zero values.
    precision : int, optional
        Decimal precision.
    """
    print(solution_summary(variables, objective_value=objective_value,
                           status=status, show_zero=show_zero, precision=precision))
