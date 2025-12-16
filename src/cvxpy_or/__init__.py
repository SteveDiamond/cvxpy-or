"""cvxpy-or: Operations Research-style modeling for CVXPY.

This package provides AMPL/Pyomo-style set-based indexing for CVXPY,
enabling natural modeling of transportation, scheduling, and other OR problems.

Example
-------
>>> from cvxpy_or import Set, Variable, Parameter, sum_by
>>> import cvxpy as cp
>>>
>>> warehouses = Set(['W1', 'W2', 'W3'], name='warehouses')
>>> customers = Set(['C1', 'C2'], name='customers')
>>> routes = Set.cross(warehouses, customers, name='routes')
>>>
>>> cost = Parameter(routes, data={('W1', 'C1'): 10, ...})
>>> ship = Variable(routes, nonneg=True)
>>>
>>> prob = cp.Problem(cp.Minimize(cost @ ship), [...])
"""

from cvxpy_or.aggregations import (
    count_by,
    group_keys,
    max_by,
    mean_by,
    min_by,
)
from cvxpy_or.constraints import (
    at_least_k,
    at_most_k,
    bounds,
    exactly_k,
    flow_balance,
    implies,
    mutex,
    one_of,
)
from cvxpy_or.display import (
    parameter_table,
    print_parameter,
    print_solution,
    print_variable,
    solution_summary,
    variable_table,
)
from cvxpy_or.io import (
    parameter_from_dataframe,
    parameter_from_series,
    parameter_to_dataframe,
    set_from_dataframe,
    set_from_index,
    set_from_series,
    variable_to_dataframe,
)
from cvxpy_or.model import Model
from cvxpy_or.sets import Parameter, Set, Variable, sum_by, where
from cvxpy_or.validation import (
    ValidationError,
    validate_bounds,
    validate_keys,
    validate_numeric,
    validate_parameter,
)

__all__ = [
    # Core
    "Set",
    "Variable",
    "Parameter",
    "Model",
    # Aggregations
    "sum_by",
    "mean_by",
    "min_by",
    "max_by",
    "count_by",
    "group_keys",
    # Filtering
    "where",
    # Constraints
    "at_most_k",
    "at_least_k",
    "exactly_k",
    "implies",
    "mutex",
    "one_of",
    "bounds",
    "flow_balance",
    # Display
    "print_variable",
    "print_parameter",
    "print_solution",
    "variable_table",
    "parameter_table",
    "solution_summary",
    # I/O
    "set_from_series",
    "set_from_dataframe",
    "set_from_index",
    "parameter_from_dataframe",
    "parameter_from_series",
    "variable_to_dataframe",
    "parameter_to_dataframe",
    # Validation
    "ValidationError",
    "validate_keys",
    "validate_numeric",
    "validate_bounds",
    "validate_parameter",
]
__version__ = "0.1.0"
