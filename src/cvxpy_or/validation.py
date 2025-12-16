"""Validation utilities for cvxpy-or.

This module provides helpful validation and error messages for Sets,
Variables, and Parameters.
"""

from __future__ import annotations

from difflib import get_close_matches
from typing import TYPE_CHECKING, Any, Hashable, Sequence

if TYPE_CHECKING:
    from cvxpy_or.sets import Parameter, Set


class ValidationError(Exception):
    """Raised when validation fails."""

    pass


def validate_keys(
    data: dict[Hashable, Any],
    index: Set,
    *,
    allow_partial: bool = False,
    context: str = "data",
) -> None:
    """Validate that data keys match an index Set.

    Parameters
    ----------
    data : dict
        The data dict to validate.
    index : Set
        The index Set to validate against.
    allow_partial : bool, optional
        If True, allow missing keys. Default is False.
    context : str, optional
        Context string for error messages (e.g., "Parameter 'cost'").

    Raises
    ------
    ValidationError
        If keys are invalid or missing.

    Examples
    --------
    >>> routes = Set([('W1', 'C1'), ('W1', 'C2'), ('W2', 'C1')])
    >>> data = {('W1', 'C1'): 10, ('TYPO', 'C1'): 20}
    >>> validate_keys(data, routes)
    ValidationError: Invalid key ('TYPO', 'C1') in data.
        Key not found in index 'routes'.
        Position 0 ('warehouses'): 'TYPO' is not valid.
        Did you mean: 'W1' or 'W2'?
    """
    index_elements = set(index._elements)
    data_keys = set(data.keys())

    # Check for extra keys (keys in data but not in index)
    extra_keys = data_keys - index_elements
    if extra_keys:
        key = next(iter(extra_keys))  # Report first invalid key
        msg = _format_invalid_key_error(key, index, context)
        raise ValidationError(msg)

    # Check for missing keys (unless partial data is allowed)
    if not allow_partial:
        missing_keys = index_elements - data_keys
        if missing_keys:
            n_missing = len(missing_keys)
            examples = list(missing_keys)[:3]
            msg = (
                f"Missing {n_missing} key(s) in {context}.\n"
                f"    Index '{index.name}' has {len(index)} elements, "
                f"but only {len(data)} provided.\n"
                f"    Missing examples: {examples}"
            )
            raise ValidationError(msg)


def _format_invalid_key_error(key: Hashable, index: Set, context: str) -> str:
    """Format a helpful error message for an invalid key."""
    lines = [f"Invalid key {key!r} in {context}."]
    lines.append(f"    Key not found in index '{index.name}'.")

    # For compound indices, identify which position is wrong
    if index._is_compound and isinstance(key, tuple):
        # Get the unique values at each position
        first_elem = index._elements[0]
        n_positions = len(first_elem)

        if len(key) != n_positions:
            lines.append(
                f"    Expected tuple of length {n_positions}, got length {len(key)}."
            )
        else:
            for pos in range(n_positions):
                valid_values = {elem[pos] for elem in index._elements}
                if key[pos] not in valid_values:
                    pos_name = (
                        f"'{index._names[pos]}'"
                        if index._names
                        else str(pos)
                    )
                    lines.append(
                        f"    Position {pos} ({pos_name}): {key[pos]!r} is not valid."
                    )

                    # Suggest similar values
                    similar = get_close_matches(
                        str(key[pos]), [str(v) for v in valid_values], n=2, cutoff=0.6
                    )
                    if similar:
                        suggestions = " or ".join(f"'{s}'" for s in similar)
                        lines.append(f"    Did you mean: {suggestions}?")
                    else:
                        examples = list(valid_values)[:5]
                        lines.append(f"    Valid values: {examples}")
    else:
        # Simple index - suggest similar elements
        similar = get_close_matches(
            str(key), [str(e) for e in index._elements], n=3, cutoff=0.6
        )
        if similar:
            suggestions = ", ".join(f"'{s}'" for s in similar)
            lines.append(f"    Did you mean: {suggestions}?")
        else:
            examples = index._elements[:5]
            lines.append(f"    Valid elements: {examples}")

    return "\n".join(lines)


def validate_numeric(
    data: dict[Hashable, Any],
    context: str = "data",
) -> None:
    """Validate that all values in data are numeric.

    Parameters
    ----------
    data : dict
        The data dict to validate.
    context : str, optional
        Context string for error messages.

    Raises
    ------
    ValidationError
        If any value is not numeric.
    """
    for key, value in data.items():
        if not isinstance(value, (int, float)):
            raise ValidationError(
                f"Non-numeric value in {context}.\n"
                f"    Key: {key!r}\n"
                f"    Value: {value!r} (type: {type(value).__name__})\n"
                f"    Expected: int or float"
            )


def validate_bounds(
    data: dict[Hashable, Any],
    *,
    lower: float | None = None,
    upper: float | None = None,
    context: str = "data",
) -> None:
    """Validate that all values are within bounds.

    Parameters
    ----------
    data : dict
        The data dict to validate.
    lower : float, optional
        Lower bound (inclusive).
    upper : float, optional
        Upper bound (inclusive).
    context : str, optional
        Context string for error messages.

    Raises
    ------
    ValidationError
        If any value is out of bounds.
    """
    for key, value in data.items():
        if lower is not None and value < lower:
            raise ValidationError(
                f"Value below lower bound in {context}.\n"
                f"    Key: {key!r}\n"
                f"    Value: {value}\n"
                f"    Lower bound: {lower}"
            )
        if upper is not None and value > upper:
            raise ValidationError(
                f"Value above upper bound in {context}.\n"
                f"    Key: {key!r}\n"
                f"    Value: {value}\n"
                f"    Upper bound: {upper}"
            )


def validate_parameter(
    param: Parameter,
    *,
    complete: bool = True,
    numeric: bool = True,
    lower: float | None = None,
    upper: float | None = None,
) -> None:
    """Validate a Parameter's data.

    Parameters
    ----------
    param : Parameter
        The Parameter to validate.
    complete : bool, optional
        Check that all index elements have values. Default True.
    numeric : bool, optional
        Check that all values are numeric. Default True.
    lower : float, optional
        Lower bound for values.
    upper : float, optional
        Upper bound for values.

    Raises
    ------
    ValidationError
        If validation fails.
    """
    if param.value is None:
        raise ValidationError(
            f"Parameter '{param.name}' has no data set.\n"
            f"    Use param.set_data({{...}}) to set values."
        )

    context = f"Parameter '{param.name}'"

    # Convert numpy array to dict for validation
    data = {
        elem: param.value[i]
        for i, elem in enumerate(param.index._elements)
    }

    if numeric:
        validate_numeric(data, context=context)

    if lower is not None or upper is not None:
        validate_bounds(data, lower=lower, upper=upper, context=context)


def suggest_key(key: Hashable, index: Set) -> str | None:
    """Suggest a similar key from the index.

    Parameters
    ----------
    key : Hashable
        The invalid key.
    index : Set
        The index to search for suggestions.

    Returns
    -------
    str | None
        The best suggestion, or None if no good match.
    """
    matches = get_close_matches(
        str(key), [str(e) for e in index._elements], n=1, cutoff=0.6
    )
    return matches[0] if matches else None
