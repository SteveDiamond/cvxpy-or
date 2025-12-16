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

from cvxpy_or.sets import Parameter, Set, Variable, sum_by, where

__all__ = ["Set", "Variable", "Parameter", "sum_by", "where"]
__version__ = "0.1.0"
