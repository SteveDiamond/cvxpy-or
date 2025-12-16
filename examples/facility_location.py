#!/usr/bin/env python3
"""Facility Location Problem using cvxpy-or.

This example demonstrates the uncapacitated facility location problem (UFLP)
using the Model class and showcasing validation features.

Problem: Choose which facilities to open and how to assign customers to
minimize total fixed + transportation costs.
"""

import cvxpy as cp

from cvxpy_or import Model, Set, sum_by, validate_keys, ValidationError

# =============================================================================
# CREATE MODEL
# =============================================================================

m = Model(name='facility_location')

# =============================================================================
# INDEX SETS
# =============================================================================

facilities = Set(
    ['Atlanta', 'Boston', 'Chicago', 'Denver', 'Seattle'],
    name='facilities'
)

customers = Set(
    ['NYC', 'LA', 'Houston', 'Phoenix', 'Dallas', 'Miami'],
    name='customers'
)

connections = Set.cross(facilities, customers, name='connections')

print(f"Potential facilities: {len(facilities)}")
print(f"Customers: {len(customers)}")
print(f"Possible connections: {len(connections)}")
print()

# =============================================================================
# PARAMETERS (with validation)
# =============================================================================

# Fixed cost to open each facility ($000s)
fixed_cost_data = {
    'Atlanta': 500,
    'Boston': 600,
    'Chicago': 550,
    'Denver': 450,
    'Seattle': 650,
}

# Validate the data matches the index (demonstrates validation feature)
try:
    validate_keys(fixed_cost_data, facilities)
    print("Fixed cost data validated successfully")
except ValidationError as e:
    print(f"Validation error: {e}")

fixed_cost = m.add_parameter(facilities, data=fixed_cost_data, name='fixed_cost')

# Transportation cost per unit from facility to customer ($/unit)
transport_cost_data = {
    # Atlanta
    ('Atlanta', 'NYC'): 15, ('Atlanta', 'LA'): 40,
    ('Atlanta', 'Houston'): 20, ('Atlanta', 'Phoenix'): 35,
    ('Atlanta', 'Dallas'): 18, ('Atlanta', 'Miami'): 12,
    # Boston
    ('Boston', 'NYC'): 8, ('Boston', 'LA'): 50,
    ('Boston', 'Houston'): 35, ('Boston', 'Phoenix'): 45,
    ('Boston', 'Dallas'): 32, ('Boston', 'Miami'): 25,
    # Chicago
    ('Chicago', 'NYC'): 18, ('Chicago', 'LA'): 35,
    ('Chicago', 'Houston'): 25, ('Chicago', 'Phoenix'): 30,
    ('Chicago', 'Dallas'): 20, ('Chicago', 'Miami'): 28,
    # Denver
    ('Denver', 'NYC'): 30, ('Denver', 'LA'): 20,
    ('Denver', 'Houston'): 18, ('Denver', 'Phoenix'): 12,
    ('Denver', 'Dallas'): 15, ('Denver', 'Miami'): 35,
    # Seattle
    ('Seattle', 'NYC'): 45, ('Seattle', 'LA'): 18,
    ('Seattle', 'Houston'): 35, ('Seattle', 'Phoenix'): 25,
    ('Seattle', 'Dallas'): 32, ('Seattle', 'Miami'): 50,
}
transport_cost = m.add_parameter(connections, data=transport_cost_data, name='transport_cost')

# Customer demand (units)
demand_data = {
    'NYC': 100, 'LA': 150, 'Houston': 80,
    'Phoenix': 60, 'Dallas': 90, 'Miami': 70,
}
demand = m.add_parameter(customers, data=demand_data, name='demand')

# Facility capacity (large enough to be effectively uncapacitated)
capacity_data = {f: 1000 for f in facilities}
capacity = m.add_parameter(facilities, data=capacity_data, name='capacity')

# =============================================================================
# DECISION VARIABLES
# =============================================================================

# Binary (relaxed): 1 if facility is opened
open_facility = m.add_variable(facilities, nonneg=True, name='open')

# Amount shipped from facility to customer
ship = m.add_variable(connections, nonneg=True, name='ship')

# =============================================================================
# CONSTRAINTS
# =============================================================================

# Demand satisfaction: each customer's demand must be met
m.add_constraint('demand', sum_by(ship, 'customers') >= demand)

# Capacity linking: can only ship from open facilities
m.add_constraint('capacity', sum_by(ship, 'facilities') <= cp.multiply(capacity, open_facility))

# Facility open variable bounds (LP relaxation)
m.add_constraint('open_bound', open_facility <= 1)

# =============================================================================
# OBJECTIVE
# =============================================================================

fixed_cost_expr = fixed_cost @ open_facility
transport_cost_expr = transport_cost @ ship
m.minimize(fixed_cost_expr + transport_cost_expr)

# =============================================================================
# SOLVE
# =============================================================================

m.solve()

# =============================================================================
# RESULTS
# =============================================================================

m.print_summary()
print()

print(f"Cost breakdown:")
print(f"  Fixed cost: ${fixed_cost_expr.value:.0f}k")
print(f"  Transport cost: ${transport_cost_expr.value:.0f}k")
print()

print("=== Facility Decisions ===")
for f in facilities:
    val = open_facility.get_value(f)
    fc = fixed_cost.get_value(f)
    if val > 0.01:
        status = "OPEN" if val > 0.99 else f"PARTIAL ({val:.2f})"
        print(f"  {f:10}: {status} (fixed cost: ${fc}k)")
    else:
        print(f"  {f:10}: CLOSED")

print()
print("=== Shipping Plan (non-zero flows) ===")
for f in facilities:
    if open_facility.get_value(f) > 0.01:
        print(f"\nFrom {f}:")
        for c in customers:
            val = ship.get_value((f, c))
            if val > 0.01:
                tc = transport_cost.get_value((f, c))
                print(f"  -> {c:10}: {val:6.1f} units @ ${tc}/unit")
