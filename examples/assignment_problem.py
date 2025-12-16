#!/usr/bin/env python3
"""Assignment Problem using cvxpy-or.

This example demonstrates the classic assignment problem:
optimally assign workers to tasks to minimize total cost.

Problem: Given n workers and n tasks, assign each worker to exactly one task
and each task to exactly one worker, minimizing total assignment cost.

Note: This is the LP relaxation. For integer-sized problems, the LP relaxation
gives integer solutions due to the total unimodularity of the constraint matrix.
"""

import cvxpy as cp

from cvxpy_or import Parameter, Set, Variable, sum_by

# =============================================================================
# INDEX SETS
# =============================================================================

# Workers available
workers = Set(
    ['Alice', 'Bob', 'Carol', 'David'],
    name='workers'
)

# Tasks to complete
tasks = Set(
    ['Task_A', 'Task_B', 'Task_C', 'Task_D'],
    name='tasks'
)

# All possible assignments (worker, task pairs)
assignments = Set.cross(workers, tasks, name='assignments')

print(f"Workers: {len(workers)}")
print(f"Tasks: {len(tasks)}")
print(f"Possible assignments: {len(assignments)}")
print()

# =============================================================================
# PARAMETERS
# =============================================================================

# Cost (or time) for each worker to complete each task
# Lower is better
cost_data = {
    # Alice is best at Task_A and Task_D
    ('Alice', 'Task_A'): 9, ('Alice', 'Task_B'): 11,
    ('Alice', 'Task_C'): 14, ('Alice', 'Task_D'): 8,

    # Bob is best at Task_B
    ('Bob', 'Task_A'): 6, ('Bob', 'Task_B'): 4,
    ('Bob', 'Task_C'): 10, ('Bob', 'Task_D'): 7,

    # Carol is best at Task_A
    ('Carol', 'Task_A'): 5, ('Carol', 'Task_B'): 8,
    ('Carol', 'Task_C'): 12, ('Carol', 'Task_D'): 11,

    # David is best at Task_C
    ('David', 'Task_A'): 7, ('David', 'Task_B'): 9,
    ('David', 'Task_C'): 3, ('David', 'Task_D'): 10,
}
cost = Parameter(assignments, data=cost_data, name='cost')

# =============================================================================
# DECISION VARIABLES
# =============================================================================

# Assignment variable: assign[w, t] = 1 if worker w assigned to task t
# Using nonneg=True for LP relaxation (will be binary due to problem structure)
assign = Variable(assignments, nonneg=True, name='assign')

# =============================================================================
# OBJECTIVE: Minimize total assignment cost
# =============================================================================

objective = cp.Minimize(cost @ assign)

# =============================================================================
# CONSTRAINTS
# =============================================================================

# 1. Each worker is assigned to exactly one task
#    sum over tasks of assign[w, t] = 1 for each worker w
#    Using sum_by to aggregate by 'workers' (summing over tasks)
worker_assignment = sum_by(assign, 'workers', index=assignments)

# 2. Each task is assigned to exactly one worker
#    sum over workers of assign[w, t] = 1 for each task t
#    Using sum_by to aggregate by 'tasks' (summing over workers)
task_assignment = sum_by(assign, 'tasks', index=assignments)

# 3. Upper bound on assignment (for LP relaxation)
#    assign[w, t] <= 1

constraints = [
    # Each worker assigned to exactly one task
    worker_assignment == 1,

    # Each task assigned to exactly one worker
    task_assignment == 1,

    # Upper bound (LP relaxation of binary constraint)
    assign <= 1,
]

# =============================================================================
# SOLVE
# =============================================================================

prob = cp.Problem(objective, constraints)
result = prob.solve()

print(f"Status: {prob.status}")
print(f"Optimal total cost: {result:.0f}")
print()

# =============================================================================
# RESULTS
# =============================================================================

print("=== Optimal Assignment ===")
for w in workers:
    for t in tasks:
        val = assign.get_value((w, t))
        if val > 0.5:  # Assigned (threshold for numerical tolerance)
            c = cost.get_value((w, t))
            print(f"  {w:8} -> {t:8} (cost: {c})")

print()
print("=== Assignment Matrix ===")
header = "         " + "  ".join(f"{t:8}" for t in tasks)
print(header)
for w in workers:
    row = f"{w:8} "
    for t in tasks:
        val = assign.get_value((w, t))
        row += f"  {val:8.2f}"
    print(row)
