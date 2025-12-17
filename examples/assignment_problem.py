#!/usr/bin/env python3
"""Assignment Problem using cvxpy-or.

This example demonstrates the classic assignment problem using:
- Model class for clean problem definition
- pandas DataFrames for data input and solution output
- Rich tables for fancy display

Problem: Assign workers to tasks to minimize total cost.
"""

import pandas as pd

from cvxpy_or import (
    Model,
    Set,
    parameter_from_dataframe,
    print_variable,
    sum_by,
    variable_to_dataframe,
)

# =============================================================================
# INPUT DATA (as pandas DataFrame)
# =============================================================================

# Cost matrix as a DataFrame (more realistic data input)
cost_df = pd.DataFrame(
    {
        "worker": [
            "Alice",
            "Alice",
            "Alice",
            "Alice",
            "Bob",
            "Bob",
            "Bob",
            "Bob",
            "Carol",
            "Carol",
            "Carol",
            "Carol",
            "David",
            "David",
            "David",
            "David",
        ],
        "task": ["Task_A", "Task_B", "Task_C", "Task_D"] * 4,
        "cost": [
            9,
            11,
            14,
            8,  # Alice
            6,
            4,
            10,
            7,  # Bob
            5,
            8,
            12,
            11,  # Carol
            7,
            9,
            3,
            10,
        ],  # David
    }
)

print("=== Input Cost Data ===")
print(cost_df.pivot(index="worker", columns="task", values="cost"))
print()

# =============================================================================
# CREATE MODEL
# =============================================================================

m = Model(name="assignment")

# Define sets from DataFrame columns
workers = Set(cost_df["worker"].unique().tolist(), name="workers")
tasks = Set(cost_df["task"].unique().tolist(), name="tasks")
assignments = Set.cross(workers, tasks, name="assignments")

# Load cost parameter directly from DataFrame
cost = parameter_from_dataframe(
    cost_df, index_cols=["worker", "task"], value_col="cost", name="cost"
)

# Decision variable
assign = m.add_variable(assignments, nonneg=True, name="assign")

# Constraints
m.add_constraint("one_task_per_worker", sum_by(assign, "workers") == 1)
m.add_constraint("one_worker_per_task", sum_by(assign, "tasks") == 1)
m.add_constraint("upper_bound", assign <= 1)

# Objective
m.minimize(cost @ assign)

# =============================================================================
# SOLVE
# =============================================================================

m.solve()

# =============================================================================
# RESULTS
# =============================================================================

print("=== Model Summary ===")
m.print_summary()
print()

# Export solution to pandas DataFrame
print("=== Solution as DataFrame ===")
solution_df = variable_to_dataframe(assign, value_col="assigned")
solution_df = solution_df[solution_df["assigned"] > 0.5]  # Filter to assignments
# Rename column to match original DataFrame for merge
solution_df = solution_df.rename(columns={"workers": "worker", "tasks": "task"})
solution_df = solution_df.merge(cost_df, on=["worker", "task"])
print(solution_df[["worker", "task", "cost"]].to_string(index=False))
print()
print(f"Total cost: {solution_df['cost'].sum()}")
print()

# Fancy table display
print("=== Assignment Matrix (Rich Table) ===")
print_variable(assign, show_zero=False, precision=0)
