#!/usr/bin/env python3
"""Multi-Period Transportation Problem using cvxpy-or.

This example demonstrates the set-based indexing API for CVXPY, which provides
AMPL/Pyomo-style modeling while using native CVXPY operations.

Problem: Ship products from warehouses to customers over multiple time periods,
minimizing total cost while respecting supply, demand, and inventory constraints.
"""

import cvxpy as cp

from cvxpy_or import Parameter, Set, Variable

# =============================================================================
# INDEX SETS
# =============================================================================

# Simple indices
warehouses = Set(['Seattle', 'Denver', 'Chicago'], name='warehouses')
customers = Set(['NYC', 'LA', 'Houston', 'Miami'], name='customers')
periods = Set(['Jan', 'Feb', 'Mar'], name='periods')

# Cross-product indices (Cartesian products)
# 2D: all warehouse-customer pairs
routes = Set.cross(warehouses, customers, name='routes')
# Names auto-generated: ('warehouses', 'customers')

# 3D: shipments indexed by (warehouse, customer, period)
shipments = Set.cross(warehouses, customers, periods, name='shipments')
# Names: ('warehouses', 'customers', 'periods')

# Inventory indexed by (warehouse, period)
inventory_idx = Set.cross(warehouses, periods, name='inventory_idx')

print(f"Routes: {len(routes)} combinations")
print(f"Shipments: {len(shipments)} combinations (3D)")
print(f"Inventory slots: {len(inventory_idx)} combinations")
print()

# =============================================================================
# PARAMETERS
# =============================================================================

# Shipping cost per unit (warehouse -> customer)
cost_data = {
    ('Seattle', 'NYC'): 2.5, ('Seattle', 'LA'): 1.0,
    ('Seattle', 'Houston'): 1.8, ('Seattle', 'Miami'): 3.0,
    ('Denver', 'NYC'): 2.0, ('Denver', 'LA'): 1.5,
    ('Denver', 'Houston'): 1.2, ('Denver', 'Miami'): 2.2,
    ('Chicago', 'NYC'): 1.0, ('Chicago', 'LA'): 2.5,
    ('Chicago', 'Houston'): 1.5, ('Chicago', 'Miami'): 1.8,
}
cost = Parameter(routes, data=cost_data, name='cost')

# Supply capacity per warehouse per period
supply_data = {
    ('Seattle', 'Jan'): 100, ('Seattle', 'Feb'): 120, ('Seattle', 'Mar'): 110,
    ('Denver', 'Jan'): 80, ('Denver', 'Feb'): 90, ('Denver', 'Mar'): 85,
    ('Chicago', 'Jan'): 150, ('Chicago', 'Feb'): 140, ('Chicago', 'Mar'): 160,
}
supply = Parameter(inventory_idx, data=supply_data, name='supply')

# Customer demand per period
demand_idx = Set.cross(customers, periods, name='demand_idx')
demand_data = {
    ('NYC', 'Jan'): 60, ('NYC', 'Feb'): 70, ('NYC', 'Mar'): 65,
    ('LA', 'Jan'): 50, ('LA', 'Feb'): 55, ('LA', 'Mar'): 60,
    ('Houston', 'Jan'): 40, ('Houston', 'Feb'): 45, ('Houston', 'Mar'): 50,
    ('Miami', 'Jan'): 30, ('Miami', 'Feb'): 35, ('Miami', 'Mar'): 40,
}
demand = Parameter(demand_idx, data=demand_data, name='demand')

# Holding cost per unit per period (at warehouses)
holding_cost_data = {'Seattle': 0.1, 'Denver': 0.08, 'Chicago': 0.12}
holding_cost = Parameter(
    warehouses,
    data=holding_cost_data,
    name='holding_cost'
)

# =============================================================================
# DECISION VARIABLES
# =============================================================================

# Shipment quantities: ship[warehouse, customer, period]
ship = Variable(shipments, nonneg=True, name='ship')

# Inventory levels: inv[warehouse, period]
inv = Variable(inventory_idx, nonneg=True, name='inv')

# =============================================================================
# OBJECTIVE: Minimize total shipping + holding cost (NO FOR LOOPS!)
# =============================================================================

# Shipping cost: cost[w,c] * ship[w,c,t] summed over all (w,c,t)
# First aggregate ship over periods to match cost's shape, then use @
# cost @ ship.sum_by(['warehouses', 'customers']) = sum_{w,c} cost[w,c] * sum_t ship[w,c,t]
#                                                 = sum_{w,c,t} cost[w,c] * ship[w,c,t]
shipping_cost_expr = cost @ ship.sum_by(['warehouses', 'customers'])

# Holding cost: holding_cost[w] * inv[w,t] summed over all (w,t)
# First aggregate inv over periods to match holding_cost's shape, then use @
holding_cost_expr = holding_cost @ inv.sum_by('warehouses')

objective = cp.Minimize(shipping_cost_expr + holding_cost_expr)

# =============================================================================
# CONSTRAINTS (NO FOR LOOPS!)
# =============================================================================

# 1. Supply constraint: total shipped from each warehouse in each period <= supply
#    sum over customers of ship[w, c, t] <= supply[w, t]
#
#    Using sum_by(['warehouses', 'periods']): keeps (w, t), sums over customers

# 2. Demand constraint: total received by each customer in each period >= demand
#    sum over warehouses of ship[w, c, t] >= demand[c, t]
#
#    Using sum_by(['customers', 'periods']): keeps (c, t), sums over warehouses

# 3. Inventory balance: inv[w, t] = supply[w, t] - sum_c ship[w, c, t]

constraints = [
    # Supply: sum over customers for each (warehouse, period)
    ship.sum_by(['warehouses', 'periods']) <= supply,

    # Demand: sum over warehouses for each (customer, period)
    ship.sum_by(['customers', 'periods']) >= demand,

    # Inventory balance: remaining supply after shipping
    inv == supply - ship.sum_by(['warehouses', 'periods']),
]

# =============================================================================
# SOLVE
# =============================================================================

prob = cp.Problem(objective, constraints)
result = prob.solve()

print(f"Status: {prob.status}")
print(f"Optimal cost: ${result:.2f}")
print()

# =============================================================================
# RESULTS
# =============================================================================

print("=== Shipment Plan ===")
for t in periods:
    print(f"\n{t}:")
    for w in warehouses:
        for c in customers:
            val = ship[(w, c, t)].value
            if val > 0.01:
                print(f"  {w:10} -> {c:10}: {val:6.1f} units")

print("\n=== Inventory Levels ===")
for t in periods:
    print(f"\n{t}:")
    for w in warehouses:
        val = inv[(w, t)].value
        print(f"  {w:10}: {val:6.1f} units")

# =============================================================================
# DEMONSTRATE SEAMLESS CVXPY INTEGRATION
# =============================================================================

print("\n" + "=" * 60)
print("SEAMLESS CVXPY INTEGRATION")
print("=" * 60)

# Pattern 1: Native @ operator for inner product
routes_only = Set.cross(warehouses, customers)
cost2 = Parameter(routes_only, data=cost_data)
ship2 = Variable(routes_only, nonneg=True)

inner_prod = cost2 @ ship2  # Native CVXPY inner product!
print(f"\n1. Inner product (cost @ ship):")
print(f"   Type: {type(inner_prod).__name__}")
print(f"   This is native CVXPY - no wrapper needed!")

# Pattern 2: sum_by returns plain cp.Expression
agg_by_warehouse = ship2.sum_by('warehouses')
print(f"\n2. sum_by('warehouses'):")
print(f"   Type: {type(agg_by_warehouse).__name__}")
print(f"   Shape: {agg_by_warehouse.shape}")

# Pattern 3: All CVXPY atoms work directly
print(f"\n3. CVXPY atoms work directly on Variable:")
print(f"   cp.abs(ship2): {type(cp.abs(ship2)).__name__}")
print(f"   cp.sum(ship2): {type(cp.sum(ship2)).__name__}")
print(f"   cp.sum_squares(ship2): {type(cp.sum_squares(ship2)).__name__}")
print(f"   cp.norm(ship2): {type(cp.norm(ship2)).__name__}")

# Pattern 4: Constraints work natively
print(f"\n4. Constraints work natively:")
print(f"   ship2 >= 0: {type(ship2 >= 0).__name__}")
print(f"   ship2 <= cost2: {type(ship2 <= cost2).__name__}")

# Pattern 5: Variable and Parameter ARE CVXPY objects
print(f"\n5. Inheritance (no wrappers!):")
print(f"   isinstance(ship2, cp.Variable): {isinstance(ship2, cp.Variable)}")
print(f"   isinstance(cost2, cp.Parameter): {isinstance(cost2, cp.Parameter)}")

# Pattern 6: Named indexing
print(f"\n6. Named indexing:")
print(f"   cost2[('Seattle', 'NYC')]: {cost2[('Seattle', 'NYC')]}")
print(f"   This returns the scalar at position 0")

# Pattern 7: Complex expressions
regularized_obj = cost2 @ ship2 + 0.1 * cp.sum_squares(ship2)
print(f"\n7. Complex expressions just work:")
print(f"   cost @ ship + 0.1 * sum_squares(ship)")
print(f"   Type: {type(regularized_obj).__name__}")
