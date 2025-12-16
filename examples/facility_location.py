#!/usr/bin/env python3
"""Facility Location Problem using cvxpy-or.

This example demonstrates the uncapacitated facility location problem (UFLP):
choose which facilities to open and how to assign customers to facilities
to minimize total fixed + transportation costs.

Problem: A company must decide which potential warehouse locations to open
and how to serve customer demand from those warehouses.

Note: This is the LP relaxation. For true MIP, use CVXPY with a MIP solver.
The LP relaxation often gives fractional solutions but provides a lower bound.
"""

import cvxpy as cp

from cvxpy_or import Parameter, Set, Variable, sum_by

# =============================================================================
# INDEX SETS
# =============================================================================

# Potential facility (warehouse) locations
facilities = Set(
    ['Atlanta', 'Boston', 'Chicago', 'Denver', 'Seattle'],
    name='facilities'
)

# Customer locations to serve
customers = Set(
    ['NYC', 'LA', 'Houston', 'Phoenix', 'Dallas', 'Miami'],
    name='customers'
)

# All facility-customer connections
connections = Set.cross(facilities, customers, name='connections')

print(f"Potential facilities: {len(facilities)}")
print(f"Customers: {len(customers)}")
print(f"Possible connections: {len(connections)}")
print()

# =============================================================================
# PARAMETERS
# =============================================================================

# Fixed cost to open each facility ($000s)
fixed_cost_data = {
    'Atlanta': 500,
    'Boston': 600,
    'Chicago': 550,
    'Denver': 450,
    'Seattle': 650,
}
fixed_cost = Parameter(facilities, data=fixed_cost_data, name='fixed_cost')

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
transport_cost = Parameter(connections, data=transport_cost_data, name='transport_cost')

# Customer demand (units)
demand_data = {
    'NYC': 100,
    'LA': 150,
    'Houston': 80,
    'Phoenix': 60,
    'Dallas': 90,
    'Miami': 70,
}
demand = Parameter(customers, data=demand_data, name='demand')

# Facility capacity (units) - large enough to be effectively uncapacitated
capacity_data = {f: 1000 for f in facilities}
capacity = Parameter(facilities, data=capacity_data, name='capacity')

# =============================================================================
# DECISION VARIABLES
# =============================================================================

# Binary (relaxed): 1 if facility is opened
# Using bounds [0, 1] for LP relaxation
open_facility = Variable(facilities, nonneg=True, name='open')

# Amount shipped from facility to customer
ship = Variable(connections, nonneg=True, name='ship')

# =============================================================================
# OBJECTIVE: Minimize fixed + transportation cost
# =============================================================================

# Fixed cost: sum of fixed_cost[f] * open[f]
fixed_cost_expr = fixed_cost @ open_facility

# Transportation cost: sum of transport_cost[f,c] * ship[f,c]
transport_cost_expr = transport_cost @ ship

objective = cp.Minimize(fixed_cost_expr + transport_cost_expr)

# =============================================================================
# CONSTRAINTS
# =============================================================================

# 1. Demand satisfaction: each customer's demand must be met
#    sum over facilities of ship[f, c] >= demand[c]
#    Using sum_by to aggregate by 'customers' (summing over facilities)
total_received = sum_by(ship, 'customers')

# 2. Capacity linking: can only ship from open facilities
#    sum over customers of ship[f, c] <= capacity[f] * open[f]
#    Using sum_by to aggregate by 'facilities' (summing over customers)
total_shipped = sum_by(ship, 'facilities')

# 3. Facility open variable bounds (LP relaxation of binary)
#    0 <= open[f] <= 1

constraints = [
    # Demand satisfaction
    total_received >= demand,

    # Capacity linking (only ship from open facilities)
    total_shipped <= cp.multiply(capacity, open_facility),

    # Open variable upper bound (LP relaxation)
    open_facility <= 1,
]

# =============================================================================
# SOLVE
# =============================================================================

prob = cp.Problem(objective, constraints)
result = prob.solve()

print(f"Status: {prob.status}")
print(f"Total cost: ${result:.0f}k")
print(f"  Fixed cost: ${fixed_cost_expr.value:.0f}k")
print(f"  Transport cost: ${transport_cost_expr.value:.0f}k")
print()

# =============================================================================
# RESULTS
# =============================================================================

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
print("=== Shipping Plan ===")
for f in facilities:
    if open_facility.get_value(f) > 0.01:
        print(f"\nFrom {f}:")
        for c in customers:
            val = ship.get_value((f, c))
            if val > 0.01:
                tc = transport_cost.get_value((f, c))
                print(f"  -> {c:10}: {val:6.1f} units (${tc}/unit)")

print()
print("=== Customer Service ===")
for c in customers:
    d = demand.get_value(c)
    received = sum(ship.get_value((f, c)) or 0 for f in facilities)
    print(f"  {c:10}: {received:6.1f} units received (demand: {d})")
