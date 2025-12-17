# Constraint Helpers

cvxpy-or provides helper functions for common constraint patterns.

## Cardinality Constraints

These constrain how many elements of a variable can be nonzero.

### at_most_k

At most k elements can be nonzero:

```python
from cvxpy_or import at_most_k

# Select at most 3 items
x = m.add_variable(items, nonneg=True, name='x')
constraints = at_most_k(x, k=3)
m.add_constraint('select_limit', constraints)
```

### exactly_k

Exactly k elements must be nonzero:

```python
from cvxpy_or import exactly_k

# Select exactly 2 facilities
open_facility = m.add_variable(facilities, boolean=True, name='open')
constraints = exactly_k(open_facility, k=2)
m.add_constraint('select_two', constraints)
```

### at_least_k

At least k elements must be nonzero:

```python
from cvxpy_or import at_least_k

# At least 1 backup server must be active
constraints = at_least_k(backup_active, k=1)
m.add_constraint('min_backup', constraints)
```

## Logical Constraints for Binary Variables

These work with binary (0/1) variables.

### implies

If a is 1, then b must be 1:

```python
from cvxpy_or import implies

# If we open a warehouse, we must hire a manager
constraint = implies(open_warehouse, hire_manager)
m.add_constraint('manager_required', constraint)
```

### mutex (Mutual Exclusion)

At most one can be 1:

```python
from cvxpy_or import mutex

# At most one option can be selected
a = m.add_variable(boolean=True)
b = m.add_variable(boolean=True)
c = m.add_variable(boolean=True)
constraint = mutex(a, b, c)
m.add_constraint('exclusive', constraint)
```

### one_of

Exactly one must be 1:

```python
from cvxpy_or import one_of

# Exactly one mode must be selected
mode_a = m.add_variable(boolean=True)
mode_b = m.add_variable(boolean=True)
mode_c = m.add_variable(boolean=True)
constraint = one_of(mode_a, mode_b, mode_c)
m.add_constraint('select_mode', constraint)
```

## Bound Constraints

### bounds

Apply lower and/or upper bounds from parameters:

```python
from cvxpy_or import bounds

# Apply bounds from parameters
lb = m.add_parameter(items, data={'A': 0, 'B': 5, 'C': 0})
ub = m.add_parameter(items, data={'A': 100, 'B': 50, 'C': 75})

constraints = bounds(x, lower=lb, upper=ub)
m.add_constraint('var_bounds', constraints)
```

Just lower bound:

```python
constraints = bounds(x, lower=lb)
```

Just upper bound:

```python
constraints = bounds(x, upper=ub)
```

## Network Flow Constraints

### flow_balance

Enforce flow conservation at nodes:

```python
from cvxpy_or import flow_balance

# Inflow - outflow = net (supply/demand)
constraints = flow_balance(
    inflow=ship_in,
    outflow=ship_out,
    net=supply_demand  # positive = supply, negative = demand
)
m.add_constraint('flow', constraints)
```

## Common Patterns

### Assignment Problem

```python
# Each worker assigned to exactly one task
m.add_constraint('worker', sum_by(assign, 'workers') == 1)

# Each task assigned to exactly one worker
m.add_constraint('task', sum_by(assign, 'tasks') == 1)

# Binary assignment
m.add_constraint('binary', assign <= 1)
```

### Facility Location

```python
# Can only ship from open facilities
# ship[f,c] <= capacity[f] * open[f]
m.add_constraint('linking',
    ship <= capacity.expand(routes, 'facilities') * open_facility.expand(routes, 'facilities'))

# Limit number of open facilities
constraints = at_most_k(open_facility, k=3)
m.add_constraint('max_facilities', constraints)
```

### Scheduling

```python
# Each time slot has at most one activity
for slot in time_slots:
    m.add_constraint(f'slot_{slot}',
        mutex(*[schedule[(activity, slot)] for activity in activities]))
```

### Knapsack

```python
# Weight limit
m.add_constraint('weight', weight @ select <= capacity)

# Select at most k items
constraints = at_most_k(select, k=5)
m.add_constraint('count', constraints)
```

## Combining with where

Filter before applying constraints:

```python
from cvxpy_or import where

# Only constrain routes from Seattle
seattle_routes = where(ship, origin='Seattle')
m.add_constraint('seattle_limit', sum_by(seattle_routes) <= 100)
```
