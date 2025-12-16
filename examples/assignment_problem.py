#!/usr/bin/env python3
"""Assignment Problem using cvxpy-or.

This example demonstrates the classic assignment problem using the Model class:
optimally assign workers to tasks to minimize total cost.

Problem: Given n workers and n tasks, assign each worker to exactly one task
and each task to exactly one worker, minimizing total assignment cost.
"""

from cvxpy_or import Model, Set, sum_by, print_variable

# =============================================================================
# CREATE MODEL
# =============================================================================

m = Model(name='assignment')

# =============================================================================
# INDEX SETS
# =============================================================================

# Workers available
m.workers = Set(['Alice', 'Bob', 'Carol', 'David'], name='workers')

# Tasks to complete
m.tasks = Set(['Task_A', 'Task_B', 'Task_C', 'Task_D'], name='tasks')

# All possible assignments (worker, task pairs)
m.assignments = Set.cross(m.workers, m.tasks, name='assignments')

print(f"Workers: {len(m.workers)}")
print(f"Tasks: {len(m.tasks)}")
print(f"Possible assignments: {len(m.assignments)}")
print()

# =============================================================================
# PARAMETERS
# =============================================================================

# Cost (or time) for each worker to complete each task
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
m.cost = m.add_parameter(m.assignments, data=cost_data, name='cost')

# =============================================================================
# DECISION VARIABLES
# =============================================================================

# Assignment variable: assign[w, t] = 1 if worker w assigned to task t
m.assign = m.add_variable(m.assignments, nonneg=True, name='assign')

# =============================================================================
# CONSTRAINTS
# =============================================================================

# Each worker is assigned to exactly one task
m.add_constraint('one_task_per_worker', sum_by(m.assign, 'workers') == 1)

# Each task is assigned to exactly one worker
m.add_constraint('one_worker_per_task', sum_by(m.assign, 'tasks') == 1)

# Upper bound on assignment (for LP relaxation)
m.add_constraint('upper_bound', m.assign <= 1)

# =============================================================================
# OBJECTIVE
# =============================================================================

m.minimize(m.cost @ m.assign)

# =============================================================================
# SOLVE
# =============================================================================

m.solve()

# =============================================================================
# RESULTS
# =============================================================================

# Print model summary
m.print_summary()
print()

# Print solution (only non-zero values)
print("=== Optimal Assignment ===")
for w in m.workers:
    for t in m.tasks:
        val = m.assign.get_value((w, t))
        if val > 0.5:  # Assigned
            c = m.cost.get_value((w, t))
            print(f"  {w:8} -> {t:8} (cost: {c})")

print()

# Use the new print_variable for a nice table view
print("=== Assignment Variable (non-zero) ===")
print_variable(m.assign, show_zero=False, precision=1)
