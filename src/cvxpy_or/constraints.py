"""Constraint builder utilities for cvxpy-or.

This module provides helper functions for common constraint patterns
in operations research problems.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import numpy as np

import cvxpy as cp

if TYPE_CHECKING:
    from cvxpy_or.sets import Parameter, Set, Variable


def at_most_k(
    var: Variable | cp.Variable,
    k: int,
    *,
    M: float = 1e6,
) -> list[cp.Constraint]:
    """Constrain at most k elements of var to be nonzero.

    This creates a cardinality constraint using Big-M formulation.
    Requires introducing binary indicator variables.

    Parameters
    ----------
    var : Variable
        The variable to constrain.
    k : int
        Maximum number of nonzero elements.
    M : float, optional
        Big-M constant. Should be larger than max possible value.

    Returns
    -------
    list[cp.Constraint]
        Constraints enforcing at most k nonzeros.

    Examples
    --------
    >>> x = Variable(items, nonneg=True)
    >>> constraints = at_most_k(x, k=3)  # Select at most 3 items
    """
    n = var.shape[0]
    z = cp.Variable(n, boolean=True, name=f"{getattr(var, 'name', 'var')}_indicator")

    constraints = [
        var <= M * z,           # If z[i]=0, var[i]=0
        cp.sum(z) <= k,         # At most k indicators are 1
    ]
    return constraints


def exactly_k(
    var: Variable | cp.Variable,
    k: int,
    *,
    M: float = 1e6,
) -> list[cp.Constraint]:
    """Constrain exactly k elements of var to be nonzero.

    Parameters
    ----------
    var : Variable
        The variable to constrain.
    k : int
        Exact number of nonzero elements.
    M : float, optional
        Big-M constant.

    Returns
    -------
    list[cp.Constraint]
        Constraints enforcing exactly k nonzeros.

    Examples
    --------
    >>> x = Variable(items, nonneg=True)
    >>> constraints = exactly_k(x, k=3)  # Select exactly 3 items
    """
    n = var.shape[0]
    z = cp.Variable(n, boolean=True, name=f"{getattr(var, 'name', 'var')}_indicator")

    constraints = [
        var <= M * z,           # If z[i]=0, var[i]=0
        cp.sum(z) == k,         # Exactly k indicators are 1
    ]
    return constraints


def at_least_k(
    var: Variable | cp.Variable,
    k: int,
    *,
    M: float = 1e6,
    epsilon: float = 1e-6,
) -> list[cp.Constraint]:
    """Constrain at least k elements of var to be nonzero.

    Parameters
    ----------
    var : Variable
        The variable to constrain.
    k : int
        Minimum number of nonzero elements.
    M : float, optional
        Big-M constant.
    epsilon : float, optional
        Small positive value to ensure strict positivity.

    Returns
    -------
    list[cp.Constraint]
        Constraints enforcing at least k nonzeros.

    Examples
    --------
    >>> x = Variable(items, nonneg=True)
    >>> constraints = at_least_k(x, k=2)  # At least 2 items selected
    """
    n = var.shape[0]
    z = cp.Variable(n, boolean=True, name=f"{getattr(var, 'name', 'var')}_indicator")

    constraints = [
        var <= M * z,           # If z[i]=0, var[i]=0
        var >= epsilon * z,     # If z[i]=1, var[i]>0
        cp.sum(z) >= k,         # At least k indicators are 1
    ]
    return constraints


def indicator(
    z: cp.Variable,
    constraint: cp.Constraint,
    *,
    M: float = 1e6,
) -> list[cp.Constraint]:
    """Create indicator constraint: if z=1, then constraint must hold.

    Uses Big-M formulation to link a binary variable to a constraint.

    Parameters
    ----------
    z : cp.Variable
        Binary indicator variable (scalar).
    constraint : cp.Constraint
        The constraint to enforce when z=1.
    M : float, optional
        Big-M constant.

    Returns
    -------
    list[cp.Constraint]
        Constraints implementing the indicator logic.

    Examples
    --------
    >>> z = cp.Variable(boolean=True)  # Decision to use route
    >>> # If z=1, then ship >= 10
    >>> constraints = indicator(z, ship[route] >= 10)

    Notes
    -----
    For inequality constraint `expr <= b`:
        - When z=1: expr <= b (constraint active)
        - When z=0: expr <= b + M (constraint relaxed)
    """
    # Extract constraint info
    # CVXPY constraints have args: constraint.args gives the expressions
    if hasattr(constraint, "args") and len(constraint.args) == 2:
        lhs = constraint.args[0]
        rhs = constraint.args[1]

        # Determine constraint type and reformulate
        if isinstance(constraint, cp.constraints.nonpos.NonPos):
            # lhs <= 0, so we want lhs <= M*(1-z)
            return [lhs <= M * (1 - z)]
        elif isinstance(constraint, cp.constraints.zero.Zero):
            # lhs == 0, need both directions
            return [
                lhs <= M * (1 - z),
                lhs >= -M * (1 - z),
            ]

    # For general inequality a @ x <= b style
    # Try to handle common patterns
    raise NotImplementedError(
        "indicator() currently supports simple inequality constraints. "
        "For complex constraints, reformulate manually with Big-M."
    )


def implies(
    a: cp.Variable,
    b: cp.Variable,
) -> cp.Constraint:
    """Create logical implication: if a=1 then b=1 (for binary variables).

    Parameters
    ----------
    a : cp.Variable
        Binary variable (antecedent).
    b : cp.Variable
        Binary variable (consequent).

    Returns
    -------
    cp.Constraint
        Constraint a <= b (if a=1, then b must be 1).

    Examples
    --------
    >>> use_route = cp.Variable(boolean=True)
    >>> open_warehouse = cp.Variable(boolean=True)
    >>> constraints = [implies(use_route, open_warehouse)]
    """
    return a <= b


def mutex(
    *vars: cp.Variable,
) -> cp.Constraint:
    """Mutual exclusion: at most one variable can be 1 (for binary variables).

    Parameters
    ----------
    *vars : cp.Variable
        Binary variables that are mutually exclusive.

    Returns
    -------
    cp.Constraint
        Constraint sum(vars) <= 1.

    Examples
    --------
    >>> # Each task can be assigned to at most one worker
    >>> assign_A = cp.Variable(boolean=True)
    >>> assign_B = cp.Variable(boolean=True)
    >>> constraints = [mutex(assign_A, assign_B)]
    """
    return cp.sum([v for v in vars]) <= 1


def one_of(
    *vars: cp.Variable,
) -> cp.Constraint:
    """Exactly one constraint: exactly one variable must be 1 (for binary variables).

    Parameters
    ----------
    *vars : cp.Variable
        Binary variables where exactly one must be selected.

    Returns
    -------
    cp.Constraint
        Constraint sum(vars) == 1.

    Examples
    --------
    >>> # Each task must be assigned to exactly one worker
    >>> assign_A = cp.Variable(boolean=True)
    >>> assign_B = cp.Variable(boolean=True)
    >>> assign_C = cp.Variable(boolean=True)
    >>> constraints = [one_of(assign_A, assign_B, assign_C)]
    """
    return cp.sum([v for v in vars]) == 1


def bounds(
    var: Variable,
    lower: Parameter | dict | float | None = None,
    upper: Parameter | dict | float | None = None,
) -> list[cp.Constraint]:
    """Apply index-specific bounds to a variable.

    Parameters
    ----------
    var : Variable
        The variable to bound.
    lower : Parameter, dict, or float, optional
        Lower bounds (per-element if dict/Parameter, scalar if float).
    upper : Parameter, dict, or float, optional
        Upper bounds (per-element if dict/Parameter, scalar if float).

    Returns
    -------
    list[cp.Constraint]
        Bound constraints.

    Examples
    --------
    >>> capacity = Parameter(routes, data={...})
    >>> ship = Variable(routes, nonneg=True)
    >>> constraints = bounds(ship, upper=capacity)
    """
    constraints = []

    if lower is not None:
        if isinstance(lower, (int, float)):
            constraints.append(var >= lower)
        else:
            # Parameter or array
            constraints.append(var >= lower)

    if upper is not None:
        if isinstance(upper, (int, float)):
            constraints.append(var <= upper)
        else:
            constraints.append(var <= upper)

    return constraints


def flow_balance(
    flow_var: Variable,
    source_pos: int | str,
    sink_pos: int | str,
    node_supply: dict | Parameter | None = None,
    node_demand: dict | Parameter | None = None,
) -> list[cp.Constraint]:
    """Create flow balance constraints for a network flow problem.

    For each node: inflow - outflow = supply - demand

    Parameters
    ----------
    flow_var : Variable
        Flow variable indexed by (source, sink) pairs.
    source_pos : int or str
        Position of source node in the index.
    sink_pos : int or str
        Position of sink node in the index.
    node_supply : dict or Parameter, optional
        Supply at each node (positive = source).
    node_demand : dict or Parameter, optional
        Demand at each node (positive = sink).

    Returns
    -------
    list[cp.Constraint]
        Flow balance constraints for each node.

    Examples
    --------
    >>> arcs = Set.cross(nodes, nodes, name='arcs')
    >>> flow = Variable(arcs, nonneg=True)
    >>> supply = {'A': 100, 'B': 0, 'C': -100}  # A source, C sink
    >>> constraints = flow_balance(flow, 'nodes', 'nodes', node_supply=supply)
    """
    from cvxpy_or.sets import sum_by

    index = flow_var._set_index

    # Get all unique nodes (from both source and sink positions)
    src_idx = index._resolve_position(source_pos)
    sink_idx = index._resolve_position(sink_pos)

    nodes = set()
    for elem in index:
        nodes.add(elem[src_idx])
        nodes.add(elem[sink_idx])

    constraints = []

    # For each node: outflow - inflow = net_supply
    for node in nodes:
        # Outflow: sum of flows where this node is source
        outflow_mask = np.array([
            1.0 if elem[src_idx] == node else 0.0
            for elem in index
        ])
        outflow = outflow_mask @ flow_var

        # Inflow: sum of flows where this node is sink
        inflow_mask = np.array([
            1.0 if elem[sink_idx] == node else 0.0
            for elem in index
        ])
        inflow = inflow_mask @ flow_var

        # Net supply
        net = 0.0
        if node_supply is not None:
            if isinstance(node_supply, dict):
                net += node_supply.get(node, 0.0)
            else:
                net += node_supply.get_value(node) or 0.0
        if node_demand is not None:
            if isinstance(node_demand, dict):
                net -= node_demand.get(node, 0.0)
            else:
                net -= node_demand.get_value(node) or 0.0

        constraints.append(outflow - inflow == net)

    return constraints
