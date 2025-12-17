# Assignment Problem

This example solves the classic assignment problem: assign workers to tasks at minimum cost.

**Source**: `examples/assignment_problem.py`

## Problem Description

A company has workers and tasks. Each worker can be assigned to any task, but each assignment has a different cost (based on skill match, distance, etc.). Each worker must be assigned to exactly one task, and each task must have exactly one worker.

**Objective**: Minimize total assignment cost.

## Mathematical Formulation

**Sets**:
- $W$ = workers
- $T$ = tasks

**Parameters**:
- $cost_{w,t}$ = cost to assign worker $w$ to task $t$

**Variables**:
- $assign_{w,t} \in [0, 1]$ = 1 if worker $w$ is assigned to task $t$

**Constraints**:
- Each worker assigned to one task: $\sum_t assign_{w,t} = 1$ for all $w$
- Each task has one worker: $\sum_w assign_{w,t} = 1$ for all $t$
- Binary (relaxed): $assign_{w,t} \leq 1$

**Objective**: Minimize $\sum_{w,t} cost_{w,t} \cdot assign_{w,t}$

## Implementation

### Data Setup

```python
import pandas as pd
from cvxpy_or import (
    Model, Set, sum_by,
    parameter_from_dataframe, variable_to_dataframe, print_variable,
)

# Cost matrix as a DataFrame
cost_df = pd.DataFrame({
    'worker': ['Alice', 'Alice', 'Alice', 'Alice',
               'Bob', 'Bob', 'Bob', 'Bob',
               'Carol', 'Carol', 'Carol', 'Carol',
               'David', 'David', 'David', 'David'],
    'task': ['Task_A', 'Task_B', 'Task_C', 'Task_D'] * 4,
    'cost': [9, 11, 14, 8,   # Alice
             6, 4, 10, 7,    # Bob
             5, 8, 12, 11,   # Carol
             7, 9, 3, 10],   # David
})

print(cost_df.pivot(index='worker', columns='task', values='cost'))
```

Output:
```
task    Task_A  Task_B  Task_C  Task_D
worker
Alice        9      11      14       8
Bob          6       4      10       7
Carol        5       8      12      11
David        7       9       3      10
```

### Model Definition

```python
m = Model(name='assignment')

# Define sets from DataFrame columns
workers = Set(cost_df['worker'].unique().tolist(), name='workers')
tasks = Set(cost_df['task'].unique().tolist(), name='tasks')
assignments = Set.cross(workers, tasks, name='assignments')

# Load cost parameter directly from DataFrame
cost = parameter_from_dataframe(
    cost_df,
    index_cols=['worker', 'task'],
    value_col='cost',
    name='cost'
)
```

### Variables and Constraints

```python
# Decision variable
assign = m.add_variable(assignments, nonneg=True, name='assign')

# Constraints
m.add_constraint('one_task_per_worker', sum_by(assign, 'workers') == 1)
m.add_constraint('one_worker_per_task', sum_by(assign, 'tasks') == 1)
m.add_constraint('upper_bound', assign <= 1)

# Objective
m.minimize(cost @ assign)
```

### Solve and Export

```python
m.solve()
m.print_summary()

# Export solution to DataFrame
solution_df = variable_to_dataframe(assign, value_col='assigned')
solution_df = solution_df[solution_df['assigned'] > 0.5]  # Filter to assignments

# Merge with cost data
solution_df = solution_df.merge(cost_df,
    left_on=['workers', 'tasks'],
    right_on=['worker', 'task'])
print(solution_df[['worker', 'task', 'cost']])
```

Output:
```
worker    task  cost
 Alice  Task_D     8
   Bob  Task_B     4
 Carol  Task_A     5
 David  Task_C     3
```

## Key Patterns

### One-to-One Assignment Constraints

```python
# Each worker gets exactly one task
m.add_constraint('worker', sum_by(assign, 'workers') == 1)

# Each task gets exactly one worker
m.add_constraint('task', sum_by(assign, 'tasks') == 1)
```

### LP Relaxation

The assignment problem has a special structure (totally unimodular) that guarantees integer solutions from the LP relaxation:

```python
# No need for boolean=True - LP relaxation gives integer solution
assign = m.add_variable(assignments, nonneg=True, name='assign')
m.add_constraint('upper_bound', assign <= 1)
```

### Alternative: Boolean Variables

For more complex assignment problems, use boolean variables:

```python
assign = m.add_variable(assignments, boolean=True, name='assign')
# No upper bound needed - boolean implies 0 or 1
```

### Filtering Solutions

```python
# Get assigned pairs (value near 1)
assigned = solution_df[solution_df['assigned'] > 0.5]

# Unassigned pairs (value near 0)
unassigned = solution_df[solution_df['assigned'] < 0.5]
```

## Variations

### Unbalanced Assignment

If |workers| ≠ |tasks|, relax the equality constraints:

```python
# Allow workers to be unassigned
m.add_constraint('worker', sum_by(assign, 'workers') <= 1)

# Or allow tasks to be unassigned
m.add_constraint('task', sum_by(assign, 'tasks') <= 1)
```

### Multiple Assignments

Allow each worker to handle up to k tasks:

```python
m.add_constraint('worker_capacity', sum_by(assign, 'workers') <= k)
```

### Skill Requirements

Some workers can't do certain tasks:

```python
from cvxpy_or import where

# Block Alice from Task_C
alice_taskc = where(assign, workers='Alice', tasks='Task_C')
m.add_constraint('skill', alice_taskc == 0)
```
