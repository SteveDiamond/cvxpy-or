#!/usr/bin/env python3
"""Facility Location Problem using cvxpy-or.

This example demonstrates:
- Loading location data from pandas DataFrames
- Geographic-style data organization
- Validation features with helpful errors
- Rich table visualization of facility decisions

Problem: Choose which facilities to open and how to assign customers to
minimize total fixed + transportation costs.
"""

import cvxpy as cp
import pandas as pd

from cvxpy_or import (
    Model,
    Set,
    ValidationError,
    parameter_from_dataframe,
    parameter_from_series,
    print_variable,
    sum_by,
    validate_keys,
    variable_to_dataframe,
)

# =============================================================================
# INPUT DATA (as pandas DataFrames)
# =============================================================================

# Facility data (fixed costs and capacities)
facility_df = pd.DataFrame(
    [
        {"facility": "Atlanta", "fixed_cost": 500, "capacity": 1000, "region": "South"},
        {"facility": "Boston", "fixed_cost": 600, "capacity": 1000, "region": "Northeast"},
        {"facility": "Chicago", "fixed_cost": 550, "capacity": 1000, "region": "Midwest"},
        {"facility": "Denver", "fixed_cost": 450, "capacity": 1000, "region": "West"},
        {"facility": "Seattle", "fixed_cost": 650, "capacity": 1000, "region": "West"},
    ]
)

# Customer data (demands)
customer_df = pd.DataFrame(
    [
        {"customer": "NYC", "demand": 100, "region": "Northeast"},
        {"customer": "LA", "demand": 150, "region": "West"},
        {"customer": "Houston", "demand": 80, "region": "South"},
        {"customer": "Phoenix", "demand": 60, "region": "West"},
        {"customer": "Dallas", "demand": 90, "region": "South"},
        {"customer": "Miami", "demand": 70, "region": "South"},
    ]
)

# Transportation costs (facility -> customer, $/unit)
transport_df = pd.DataFrame(
    [
        # From Atlanta
        {"facility": "Atlanta", "customer": "NYC", "cost": 15},
        {"facility": "Atlanta", "customer": "LA", "cost": 40},
        {"facility": "Atlanta", "customer": "Houston", "cost": 20},
        {"facility": "Atlanta", "customer": "Phoenix", "cost": 35},
        {"facility": "Atlanta", "customer": "Dallas", "cost": 18},
        {"facility": "Atlanta", "customer": "Miami", "cost": 12},
        # From Boston
        {"facility": "Boston", "customer": "NYC", "cost": 8},
        {"facility": "Boston", "customer": "LA", "cost": 50},
        {"facility": "Boston", "customer": "Houston", "cost": 35},
        {"facility": "Boston", "customer": "Phoenix", "cost": 45},
        {"facility": "Boston", "customer": "Dallas", "cost": 32},
        {"facility": "Boston", "customer": "Miami", "cost": 25},
        # From Chicago
        {"facility": "Chicago", "customer": "NYC", "cost": 18},
        {"facility": "Chicago", "customer": "LA", "cost": 35},
        {"facility": "Chicago", "customer": "Houston", "cost": 25},
        {"facility": "Chicago", "customer": "Phoenix", "cost": 30},
        {"facility": "Chicago", "customer": "Dallas", "cost": 20},
        {"facility": "Chicago", "customer": "Miami", "cost": 28},
        # From Denver
        {"facility": "Denver", "customer": "NYC", "cost": 30},
        {"facility": "Denver", "customer": "LA", "cost": 20},
        {"facility": "Denver", "customer": "Houston", "cost": 18},
        {"facility": "Denver", "customer": "Phoenix", "cost": 12},
        {"facility": "Denver", "customer": "Dallas", "cost": 15},
        {"facility": "Denver", "customer": "Miami", "cost": 35},
        # From Seattle
        {"facility": "Seattle", "customer": "NYC", "cost": 45},
        {"facility": "Seattle", "customer": "LA", "cost": 18},
        {"facility": "Seattle", "customer": "Houston", "cost": 35},
        {"facility": "Seattle", "customer": "Phoenix", "cost": 25},
        {"facility": "Seattle", "customer": "Dallas", "cost": 32},
        {"facility": "Seattle", "customer": "Miami", "cost": 50},
    ]
)

print("=== Input Data ===")
print("\nFacility Information:")
print(facility_df.set_index("facility"))
print("\nCustomer Demand:")
print(customer_df.set_index("customer"))
print("\nTransportation Costs ($/unit):")
print(transport_df.pivot(index="facility", columns="customer", values="cost"))
print()

# =============================================================================
# CREATE MODEL
# =============================================================================

m = Model(name="facility_location")

# Define sets from DataFrames
facilities = Set(facility_df["facility"].tolist(), name="facilities")
customers = Set(customer_df["customer"].tolist(), name="customers")
connections = Set.cross(facilities, customers, name="connections")

print(f"Potential facilities: {len(facilities)}")
print(f"Customers: {len(customers)}")
print(f"Possible connections: {len(connections)}")
print()

# =============================================================================
# PARAMETERS (with validation)
# =============================================================================

# Fixed cost to open each facility ($000s)
fixed_cost_data = facility_df.set_index("facility")["fixed_cost"].to_dict()

# Validate the data matches the index (demonstrates validation feature)
try:
    validate_keys(fixed_cost_data, facilities)
    print("Fixed cost data validated successfully")
except ValidationError as e:
    print(f"Validation error: {e}")

fixed_cost = parameter_from_series(
    facility_df.set_index("facility")["fixed_cost"], name="fixed_cost"
)

capacity = parameter_from_series(facility_df.set_index("facility")["capacity"], name="capacity")

transport_cost = parameter_from_dataframe(
    transport_df, index_cols=["facility", "customer"], value_col="cost", name="transport_cost"
)

demand = parameter_from_series(customer_df.set_index("customer")["demand"], name="demand")

# =============================================================================
# DECISION VARIABLES
# =============================================================================

# Binary (relaxed): 1 if facility is opened
open_facility = m.add_variable(facilities, nonneg=True, name="open")

# Amount shipped from facility to customer
ship = m.add_variable(connections, nonneg=True, name="ship")

# =============================================================================
# CONSTRAINTS
# =============================================================================

# Demand satisfaction: each customer's demand must be met
m.add_constraint("demand", sum_by(ship, "customers") >= demand)

# Capacity linking: can only ship from open facilities
m.add_constraint("capacity", sum_by(ship, "facilities") <= cp.multiply(capacity, open_facility))

# Facility open variable bounds (LP relaxation)
m.add_constraint("open_bound", open_facility <= 1)

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

print("\n=== Model Summary ===")
m.print_summary()
print()

print("=== Cost Breakdown ===")
print(f"  Fixed cost:     ${fixed_cost_expr.value:,.0f}k")
print(f"  Transport cost: ${transport_cost_expr.value:,.0f}k")
print(f"  Total cost:     ${m.value:,.0f}k")
print()

# Export facility decisions to DataFrame
print("=== Facility Decisions (as DataFrame) ===")
facility_results = variable_to_dataframe(open_facility, value_col="open_level")
facility_results["open_level"] = facility_results["open_level"].round(3)
facility_results["fixed_cost"] = facility_results["facilities"].map(
    facility_df.set_index("facility")["fixed_cost"]
)
facility_results["region"] = facility_results["facilities"].map(
    facility_df.set_index("facility")["region"]
)
facility_results["status"] = facility_results["open_level"].apply(
    lambda x: "OPEN" if x > 0.99 else ("PARTIAL" if x > 0.01 else "CLOSED")
)
print(facility_results.to_string(index=False))
print()

# Export shipping plan to DataFrame
print("=== Shipping Plan (non-zero flows) ===")
ship_df = variable_to_dataframe(ship, value_col="units")
ship_df["units"] = ship_df["units"].round(1)
ship_df = ship_df[ship_df["units"] > 0.01]

if not ship_df.empty:
    # Add cost info
    ship_df = ship_df.merge(
        transport_df, left_on=["facilities", "customers"], right_on=["facility", "customer"]
    )
    ship_df["shipping_cost"] = (ship_df["units"] * ship_df["cost"]).round(2)
    ship_df = ship_df[["facilities", "customers", "units", "cost", "shipping_cost"]]
    print(ship_df.to_string(index=False))
    print(f"\nTotal units shipped: {ship_df['units'].sum():.0f}")
    print(f"Total shipping cost: ${ship_df['shipping_cost'].sum():,.0f}k")
print()

# Summary by facility
print("=== Summary by Facility ===")
if not ship_df.empty:
    facility_summary = (
        ship_df.groupby("facilities").agg({"units": "sum", "shipping_cost": "sum"}).round(1)
    )
    facility_summary = facility_summary.reset_index()
    facility_summary.columns = ["facility", "total_units", "total_shipping_cost"]
    print(facility_summary.to_string(index=False))
print()

# Summary by customer
print("=== Summary by Customer ===")
if not ship_df.empty:
    customer_summary = (
        ship_df.groupby("customers").agg({"units": "sum", "shipping_cost": "sum"}).round(1)
    )
    customer_summary = customer_summary.reset_index()
    customer_summary.columns = ["customer", "total_units", "total_shipping_cost"]
    # Add demand for comparison
    customer_summary["demand"] = customer_summary["customer"].map(
        customer_df.set_index("customer")["demand"]
    )
    print(customer_summary.to_string(index=False))
print()

# Rich table display
print("=== Facility Open Levels (Rich Table) ===")
print_variable(open_facility, precision=3)

print("\n=== Shipping Plan (Rich Table - non-zero) ===")
print_variable(ship, show_zero=False, precision=1)
