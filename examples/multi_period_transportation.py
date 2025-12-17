#!/usr/bin/env python3
"""Multi-Period Transportation Problem using cvxpy-or.

This example demonstrates:
- Loading data from pandas DataFrames
- Multi-dimensional indexing (warehouse × customer × period)
- Exporting solutions back to DataFrames for analysis
- Rich table visualization

Problem: Ship products from warehouses to customers over multiple time periods,
minimizing total cost while respecting supply, demand, and inventory constraints.
"""

import pandas as pd

from cvxpy_or import (
    Model,
    Set,
    parameter_from_dataframe,
    parameter_from_series,
    print_variable,
    sum_by,
    variable_to_dataframe,
)

# =============================================================================
# INPUT DATA (as pandas DataFrames)
# =============================================================================

# Shipping cost per unit (warehouse -> customer)
cost_df = pd.DataFrame(
    [
        {"warehouse": "Seattle", "customer": "NYC", "cost": 2.5},
        {"warehouse": "Seattle", "customer": "LA", "cost": 1.0},
        {"warehouse": "Seattle", "customer": "Houston", "cost": 1.8},
        {"warehouse": "Seattle", "customer": "Miami", "cost": 3.0},
        {"warehouse": "Denver", "customer": "NYC", "cost": 2.0},
        {"warehouse": "Denver", "customer": "LA", "cost": 1.5},
        {"warehouse": "Denver", "customer": "Houston", "cost": 1.2},
        {"warehouse": "Denver", "customer": "Miami", "cost": 2.2},
        {"warehouse": "Chicago", "customer": "NYC", "cost": 1.0},
        {"warehouse": "Chicago", "customer": "LA", "cost": 2.5},
        {"warehouse": "Chicago", "customer": "Houston", "cost": 1.5},
        {"warehouse": "Chicago", "customer": "Miami", "cost": 1.8},
    ]
)

# Supply capacity per warehouse per period
supply_df = pd.DataFrame(
    [
        {"warehouse": "Seattle", "period": "Jan", "supply": 100},
        {"warehouse": "Seattle", "period": "Feb", "supply": 120},
        {"warehouse": "Seattle", "period": "Mar", "supply": 110},
        {"warehouse": "Denver", "period": "Jan", "supply": 80},
        {"warehouse": "Denver", "period": "Feb", "supply": 90},
        {"warehouse": "Denver", "period": "Mar", "supply": 85},
        {"warehouse": "Chicago", "period": "Jan", "supply": 150},
        {"warehouse": "Chicago", "period": "Feb", "supply": 140},
        {"warehouse": "Chicago", "period": "Mar", "supply": 160},
    ]
)

# Customer demand per period
demand_df = pd.DataFrame(
    [
        {"customer": "NYC", "period": "Jan", "demand": 60},
        {"customer": "NYC", "period": "Feb", "demand": 70},
        {"customer": "NYC", "period": "Mar", "demand": 65},
        {"customer": "LA", "period": "Jan", "demand": 50},
        {"customer": "LA", "period": "Feb", "demand": 55},
        {"customer": "LA", "period": "Mar", "demand": 60},
        {"customer": "Houston", "period": "Jan", "demand": 40},
        {"customer": "Houston", "period": "Feb", "demand": 45},
        {"customer": "Houston", "period": "Mar", "demand": 50},
        {"customer": "Miami", "period": "Jan", "demand": 30},
        {"customer": "Miami", "period": "Feb", "demand": 35},
        {"customer": "Miami", "period": "Mar", "demand": 40},
    ]
)

# Holding cost per warehouse
holding_cost_series = pd.Series(
    {"Seattle": 0.1, "Denver": 0.08, "Chicago": 0.12}, name="holding_cost"
)

print("=== Input Data ===")
print("\nCost Matrix:")
print(cost_df.pivot(index="warehouse", columns="customer", values="cost"))
print("\nSupply by Period:")
print(supply_df.pivot(index="warehouse", columns="period", values="supply"))
print("\nDemand by Period:")
print(demand_df.pivot(index="customer", columns="period", values="demand"))
print()

# =============================================================================
# CREATE MODEL
# =============================================================================

m = Model(name="multi_period_transport")

# Define sets from DataFrames
warehouses = Set(cost_df["warehouse"].unique().tolist(), name="warehouses")
customers = Set(cost_df["customer"].unique().tolist(), name="customers")
periods = Set(["Jan", "Feb", "Mar"], name="periods")

# Cross-product indices
routes = Set.cross(warehouses, customers, name="routes")
shipments = Set.cross(warehouses, customers, periods, name="shipments")
inventory_idx = Set.cross(warehouses, periods, name="inventory_idx")
demand_idx = Set.cross(customers, periods, name="demand_idx")

print(f"Routes: {len(routes)}, Shipments: {len(shipments)}, Inventory slots: {len(inventory_idx)}")
print()

# =============================================================================
# PARAMETERS (loaded from DataFrames)
# =============================================================================

cost = parameter_from_dataframe(cost_df, ["warehouse", "customer"], "cost", name="cost")
supply = parameter_from_dataframe(supply_df, ["warehouse", "period"], "supply", name="supply")
demand = parameter_from_dataframe(demand_df, ["customer", "period"], "demand", name="demand")
holding_cost = parameter_from_series(holding_cost_series, name="holding_cost")

# =============================================================================
# DECISION VARIABLES
# =============================================================================

ship = m.add_variable(shipments, nonneg=True, name="ship")
inv = m.add_variable(inventory_idx, nonneg=True, name="inventory")

# =============================================================================
# CONSTRAINTS
# =============================================================================

# Supply constraint
m.add_constraint("supply", sum_by(ship, ["warehouses", "periods"]) <= supply)

# Demand constraint
m.add_constraint("demand", sum_by(ship, ["customers", "periods"]) >= demand)

# Inventory balance
m.add_constraint("inv_balance", inv == supply - sum_by(ship, ["warehouses", "periods"]))

# =============================================================================
# OBJECTIVE
# =============================================================================

shipping_cost = cost @ sum_by(ship, ["warehouses", "customers"])
holding_cost_expr = holding_cost @ sum_by(inv, "warehouses")
m.minimize(shipping_cost + holding_cost_expr)

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

print(f"Shipping cost: ${shipping_cost.value:.2f}")
print(f"Holding cost: ${holding_cost_expr.value:.2f}")
print()

# Export shipments to DataFrame for analysis
print("=== Shipment Plan (as DataFrame) ===")
ship_df = variable_to_dataframe(ship, value_col="quantity")
ship_df = ship_df[ship_df["quantity"] > 0.01]  # Non-zero shipments
ship_df["quantity"] = ship_df["quantity"].round(1)
print(ship_df.to_string(index=False))
print()

# Pivot for better visualization
print("=== Shipments by Period ===")
for period in periods:
    period_df = ship_df[ship_df["periods"] == period]
    if not period_df.empty:
        pivot = period_df.pivot(index="warehouses", columns="customers", values="quantity").fillna(
            0
        )
        print(f"\n{period}:")
        print(pivot)

# Export inventory to DataFrame
print("\n=== Inventory Levels (as DataFrame) ===")
inv_df = variable_to_dataframe(inv, value_col="inventory")
inv_df["inventory"] = inv_df["inventory"].round(1)
pivot_inv = inv_df.pivot(index="warehouses", columns="periods", values="inventory")
print(pivot_inv)
print()

# Rich table display
print("=== Shipments (Rich Table - non-zero) ===")
print_variable(ship, show_zero=False, precision=1)
