# cvxpy-or Development Guide

## Project Overview

cvxpy-or provides AMPL/Pyomo-style set-based indexing for CVXPY, enabling natural modeling of transportation, scheduling, assignment, and other Operations Research problems.

## Architecture

```
src/cvxpy_or/
├── __init__.py      # Public API exports
├── sets.py          # Core: Set, Variable, Parameter, sum_by, where
├── model.py         # Model class (problem builder)
├── aggregations.py  # mean_by, min_by, max_by, count_by, group_keys
├── constraints.py   # Constraint helpers (at_most_k, implies, etc.)
├── display.py       # Rich table printing
├── io.py            # pandas DataFrame I/O
└── validation.py    # Data validation utilities
```

### Key Design Decisions

1. **Native CVXPY**: `Variable` and `Parameter` inherit from CVXPY classes, ensuring all CVXPY operations work seamlessly.

2. **Set-based indexing**: Elements are accessed by meaningful keys (e.g., `('Seattle', 'NYC')`) rather than numeric indices.

3. **Position-based aggregation**: `sum_by(expr, 'dimension')` groups by dimension name, not position number.

4. **Lazy evaluation**: Constraints and objectives build CVXPY expressions that are evaluated at solve time.

## Development Commands

```bash
# Install all dependencies
uv sync --all-extras

# Run tests
uv run pytest tests/ -v

# Run tests with coverage
uv run pytest tests/ --cov=cvxpy_or --cov-report=html

# Run linter
uv run ruff check src/ tests/

# Format code
uv run ruff format src/ tests/

# Build documentation
uv run sphinx-build -b html docs docs/_build/html

# Serve docs locally
uv run python -m http.server -d docs/_build/html 8000
```

## Testing Patterns

Tests are in `tests/` and use pytest:

```python
def test_sum_by_groups_correctly():
    """Test that sum_by aggregates over the correct dimension."""
    warehouses = Set(['W1', 'W2'], name='warehouses')
    customers = Set(['C1', 'C2'], name='customers')
    routes = Set.cross(warehouses, customers)

    ship = Variable(routes, nonneg=True)
    result = sum_by(ship, 'warehouses')

    # Result should be indexed by warehouses only
    assert result.index == warehouses
```

Run specific test file:
```bash
uv run pytest tests/test_aggregations.py -v
```

## Code Style

- Use type hints (`from __future__ import annotations`)
- Google-style docstrings with Examples sections
- 100 character line length
- Imports sorted with isort (via ruff)

Example function signature:
```python
def sum_by(
    expr: cp.Expression,
    positions: str | list[str] | None = None,
) -> cp.Expression:
    """Sum expression elements by grouping on specified index positions.

    Parameters
    ----------
    expr : cp.Expression
        Expression with an associated Set index.
    positions : str or list of str, optional
        Position name(s) to group by. If None, sums all elements.

    Returns
    -------
    cp.Expression
        Expression indexed by the specified positions.

    Examples
    --------
    >>> ship = Variable(routes, nonneg=True)
    >>> total_by_warehouse = sum_by(ship, 'warehouses')
    """
```

## Common Tasks

### Adding a new aggregation function

1. Add function to `src/cvxpy_or/aggregations.py`
2. Export from `src/cvxpy_or/__init__.py`
3. Add tests to `tests/test_aggregations.py`
4. Document in `docs/guide/aggregations.md`
5. Add to API reference in `docs/api/aggregations.rst`

### Adding a new constraint helper

1. Add function to `src/cvxpy_or/constraints.py`
2. Export from `src/cvxpy_or/__init__.py`
3. Add tests to `tests/test_constraints.py` (create if needed)
4. Document in `docs/guide/constraints.md`
5. Add to API reference in `docs/api/constraints.rst`

### Adding a new I/O function

1. Add function to `src/cvxpy_or/pandas_io.py` (pandas) or `src/cvxpy_or/xarray_io.py` (xarray)
2. Export from `src/cvxpy_or/__init__.py`
3. Add tests with appropriate fixtures
4. Document in `docs/guide/pandas-io.md` or `docs/guide/xarray-io.md`

## CI/CD

- **CI** (`.github/workflows/ci.yml`): Runs tests on Python 3.9-3.12 and linting on every push/PR
- **Docs** (`.github/workflows/docs.yml`): Builds and deploys docs to GitHub Pages on push to main

## Dependencies

- **Required**: cvxpy>=1.4, numpy, scipy
- **Optional**: pandas>=1.5 (I/O), rich>=13.0 (display)
- **Dev**: pytest, pytest-cov, ruff
- **Docs**: sphinx, furo, myst-parser, sphinx-copybutton, sphinx-design
