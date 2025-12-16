"""Model class for cvxpy-or.

The Model class provides a clean interface for building optimization
problems with set-based indexing.
"""

from __future__ import annotations

from typing import Any, Callable, Hashable, Sequence

import cvxpy as cp

from cvxpy_or.display import print_variable, solution_summary, variable_table
from cvxpy_or.sets import Parameter, Set, Variable, sum_by, where


class Model:
    """A wrapper for building CVXPY optimization problems.

    The Model class provides a convenient interface for:
    - Creating sets, variables, and parameters
    - Adding named constraints
    - Setting objectives
    - Solving and inspecting results

    Examples
    --------
    >>> from cvxpy_or import Model, Set
    >>>
    >>> m = Model(name='transportation')
    >>>
    >>> # Define sets
    >>> m.warehouses = Set(['W1', 'W2', 'W3'], name='warehouses')
    >>> m.customers = Set(['C1', 'C2'], name='customers')
    >>> m.routes = Set.cross(m.warehouses, m.customers)
    >>>
    >>> # Define data
    >>> m.cost = m.add_parameter(m.routes, data={...})
    >>> m.supply = m.add_parameter(m.warehouses, data={...})
    >>> m.demand = m.add_parameter(m.customers, data={...})
    >>>
    >>> # Define variables
    >>> m.ship = m.add_variable(m.routes, nonneg=True, name='ship')
    >>>
    >>> # Add constraints
    >>> m.add_constraint('supply', sum_by(m.ship, 'warehouses') <= m.supply)
    >>> m.add_constraint('demand', sum_by(m.ship, 'customers') >= m.demand)
    >>>
    >>> # Set objective
    >>> m.minimize(m.cost @ m.ship)
    >>>
    >>> # Solve
    >>> m.solve()
    >>> print(m.status)
    >>> m.print_solution()
    """

    def __init__(self, name: str | None = None):
        """Initialize a new Model.

        Parameters
        ----------
        name : str, optional
            Name for this model.
        """
        self._name = name or "model"
        self._variables: dict[str, Variable] = {}
        self._parameters: dict[str, Parameter] = {}
        self._constraints: dict[str, list[cp.Constraint]] = {}
        self._objective: cp.Expression | None = None
        self._sense: str | None = None  # 'minimize' or 'maximize'
        self._problem: cp.Problem | None = None
        self._status: str | None = None
        self._value: float | None = None

    @property
    def name(self) -> str:
        """Model name."""
        return self._name

    @property
    def status(self) -> str | None:
        """Solver status after solve()."""
        return self._status

    @property
    def value(self) -> float | None:
        """Objective value after solve()."""
        return self._value

    @property
    def objective(self) -> cp.Expression | None:
        """The objective expression."""
        return self._objective

    @property
    def variables(self) -> dict[str, Variable]:
        """Dictionary of named variables."""
        return self._variables

    @property
    def parameters(self) -> dict[str, Parameter]:
        """Dictionary of named parameters."""
        return self._parameters

    @property
    def constraints(self) -> dict[str, list[cp.Constraint]]:
        """Dictionary of named constraint groups."""
        return self._constraints

    def add_variable(
        self,
        index: Set,
        *,
        name: str | None = None,
        nonneg: bool = False,
        **kwargs,
    ) -> Variable:
        """Add a variable to the model.

        Parameters
        ----------
        index : Set
            The index set for this variable.
        name : str, optional
            Name for the variable.
        nonneg : bool, optional
            If True, constrain variable to be non-negative.
        **kwargs
            Additional arguments passed to Variable.

        Returns
        -------
        Variable
            The created variable.

        Examples
        --------
        >>> ship = m.add_variable(routes, nonneg=True, name='ship')
        """
        var = Variable(index, nonneg=nonneg, name=name, **kwargs)
        if name:
            self._variables[name] = var
        return var

    def add_parameter(
        self,
        index: Set,
        data: dict[Hashable, float] | None = None,
        *,
        name: str | None = None,
        **kwargs,
    ) -> Parameter:
        """Add a parameter to the model.

        Parameters
        ----------
        index : Set
            The index set for this parameter.
        data : dict, optional
            Initial data as {key: value}.
        name : str, optional
            Name for the parameter.
        **kwargs
            Additional arguments passed to Parameter.

        Returns
        -------
        Parameter
            The created parameter.

        Examples
        --------
        >>> cost = m.add_parameter(routes, data={('W1','C1'): 10, ...})
        """
        param = Parameter(index, data=data, name=name, **kwargs)
        if name:
            self._parameters[name] = param
        return param

    def add_constraint(
        self,
        name: str,
        constraint: cp.Constraint | list[cp.Constraint],
    ) -> None:
        """Add a named constraint or list of constraints.

        Parameters
        ----------
        name : str
            Name for this constraint group.
        constraint : cp.Constraint or list
            The constraint(s) to add.

        Examples
        --------
        >>> m.add_constraint('supply', sum_by(ship, 'warehouse') <= supply)
        >>> m.add_constraint('demand', sum_by(ship, 'customer') >= demand)
        """
        if isinstance(constraint, list):
            self._constraints[name] = constraint
        else:
            self._constraints[name] = [constraint]

    def add_constraints(
        self,
        name: str,
        constraints: list[cp.Constraint],
    ) -> None:
        """Add multiple constraints under a name.

        Alias for add_constraint() with a list.
        """
        self._constraints[name] = constraints

    def minimize(self, expr: cp.Expression) -> None:
        """Set the objective to minimize.

        Parameters
        ----------
        expr : cp.Expression
            The expression to minimize.

        Examples
        --------
        >>> m.minimize(cost @ ship)
        """
        self._objective = expr
        self._sense = "minimize"

    def maximize(self, expr: cp.Expression) -> None:
        """Set the objective to maximize.

        Parameters
        ----------
        expr : cp.Expression
            The expression to maximize.

        Examples
        --------
        >>> m.maximize(profit @ sales)
        """
        self._objective = expr
        self._sense = "maximize"

    def _build_problem(self) -> cp.Problem:
        """Build the CVXPY Problem from model components."""
        # Collect all constraints
        all_constraints = []
        for constraint_list in self._constraints.values():
            all_constraints.extend(constraint_list)

        # Build objective
        if self._objective is None:
            objective = cp.Minimize(0)  # Feasibility problem
        elif self._sense == "minimize":
            objective = cp.Minimize(self._objective)
        else:
            objective = cp.Maximize(self._objective)

        return cp.Problem(objective, all_constraints)

    def solve(self, **kwargs) -> str:
        """Solve the optimization problem.

        Parameters
        ----------
        **kwargs
            Arguments passed to CVXPY's solve().

        Returns
        -------
        str
            Solver status.

        Examples
        --------
        >>> status = m.solve()
        >>> status = m.solve(solver=cp.GUROBI, verbose=True)
        """
        self._problem = self._build_problem()
        self._problem.solve(**kwargs)
        self._status = self._problem.status
        self._value = self._problem.value
        return self._status

    def summary(self) -> str:
        """Return a summary of the model.

        Returns
        -------
        str
            Model summary string.
        """
        lines = [
            f"Model: {self._name}",
            "=" * 40,
        ]

        # Status and objective
        if self._status:
            lines.append(f"Status: {self._status}")
        if self._value is not None:
            lines.append(f"Objective: {self._value:.6g}")

        # Variables
        n_vars = sum(len(v.index) for v in self._variables.values())
        var_names = ", ".join(f"{k}({len(v.index)})" for k, v in self._variables.items())
        lines.append(f"Variables: {n_vars} ({var_names})")

        # Constraints
        n_constraints = sum(len(c) for c in self._constraints.values())
        const_names = ", ".join(f"{k}({len(c)})" for k, c in self._constraints.items())
        lines.append(f"Constraints: {n_constraints} ({const_names})")

        # Parameters
        n_params = sum(len(p.index) for p in self._parameters.values())
        if n_params > 0:
            param_names = ", ".join(f"{k}({len(p.index)})" for k, p in self._parameters.items())
            lines.append(f"Parameters: {n_params} ({param_names})")

        return "\n".join(lines)

    def print_summary(self) -> None:
        """Print a summary of the model."""
        print(self.summary())

    def print_solution(
        self,
        *,
        show_zero: bool = False,
        precision: int = 4,
    ) -> None:
        """Print the solution values for all variables.

        Parameters
        ----------
        show_zero : bool, optional
            Whether to show zero values. Default False.
        precision : int, optional
            Decimal precision. Default 4.
        """
        print(solution_summary(
            list(self._variables.values()),
            objective_value=self._value,
            status=self._status,
            show_zero=show_zero,
            precision=precision,
        ))

    def get_variable(self, name: str) -> Variable:
        """Get a variable by name.

        Parameters
        ----------
        name : str
            Variable name.

        Returns
        -------
        Variable
            The variable.

        Raises
        ------
        KeyError
            If variable not found.
        """
        if name not in self._variables:
            raise KeyError(f"Variable '{name}' not found. "
                          f"Available: {list(self._variables.keys())}")
        return self._variables[name]

    def get_parameter(self, name: str) -> Parameter:
        """Get a parameter by name.

        Parameters
        ----------
        name : str
            Parameter name.

        Returns
        -------
        Parameter
            The parameter.

        Raises
        ------
        KeyError
            If parameter not found.
        """
        if name not in self._parameters:
            raise KeyError(f"Parameter '{name}' not found. "
                          f"Available: {list(self._parameters.keys())}")
        return self._parameters[name]

    def to_dataframe(self, var_name: str | None = None):
        """Export variable solution(s) to pandas DataFrame.

        Parameters
        ----------
        var_name : str, optional
            Specific variable to export. If None, exports all.

        Returns
        -------
        pd.DataFrame
            Solution values.

        Raises
        ------
        ImportError
            If pandas is not installed.
        """
        from cvxpy_or.io import variable_to_dataframe

        if var_name:
            return variable_to_dataframe(self._variables[var_name])

        # Export all variables
        import pandas as pd

        dfs = []
        for name, var in self._variables.items():
            if var.value is not None:
                df = variable_to_dataframe(var, value_col=name)
                dfs.append(df)

        if not dfs:
            raise ValueError("No solved variables to export")

        # For multiple variables, we'd need to merge carefully
        # For simplicity, just return the first or concatenate
        if len(dfs) == 1:
            return dfs[0]

        return pd.concat(dfs, axis=1)

    def __repr__(self) -> str:
        n_vars = len(self._variables)
        n_constraints = len(self._constraints)
        return f"Model(name={self._name!r}, variables={n_vars}, constraints={n_constraints})"
