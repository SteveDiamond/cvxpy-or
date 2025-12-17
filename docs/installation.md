# Installation

## Requirements

- Python 3.9 or higher
- CVXPY 1.4 or higher
- pandas 1.5 or higher
- rich 13.0 or higher

## Install with pip

```bash
pip install cvxpy-or
```

## Install with uv

```bash
uv add cvxpy-or
```

## Optional Dependencies

### xarray Integration

For matrix-style I/O with xarray DataArrays:

```bash
pip install cvxpy-or[xarray]
```

## Development Installation

To install for development with all dependencies:

```bash
git clone https://github.com/YOUR_USERNAME/cvxpy-or.git
cd cvxpy-or
pip install -e ".[dev]"
```

Or with uv:

```bash
git clone https://github.com/YOUR_USERNAME/cvxpy-or.git
cd cvxpy-or
uv sync --all-extras
```

## Verifying Installation

```python
import cvxpy_or
print(cvxpy_or.__version__)
```

```python
from cvxpy_or import Set, Variable

# Create a simple set and variable
items = Set(['a', 'b', 'c'], name='items')
x = Variable(items, nonneg=True, name='x')
print(f"Created variable with {len(items)} elements")
```

## Solver Requirements

cvxpy-or uses CVXPY for solving optimization problems. CVXPY comes with open-source solvers (CLARABEL, SCS, OSQP) that handle most problems. For specific problem types, you may want to install additional solvers:

| Solver | Problem Types | Installation |
|--------|--------------|--------------|
| CLARABEL | LP, QP, SOCP, SDP | Included with CVXPY |
| SCS | LP, QP, SOCP, SDP | Included with CVXPY |
| OSQP | QP | Included with CVXPY |
| HiGHS | LP, MIP | Included with CVXPY |
| MOSEK | LP, QP, SOCP, SDP, MIP | Requires license |
| GUROBI | LP, QP, MIP | Requires license |

See the [CVXPY documentation](https://www.cvxpy.org/install/index.html) for more details on solver installation.
