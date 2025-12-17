#!/usr/bin/env python3
"""Diet Problem using cvxpy-or.

This example demonstrates:
- Loading nutritional data from pandas DataFrames
- Matrix constraints from DataFrame pivot tables
- Rich table display for results
- Solution export to DataFrames

Problem: Select foods to purchase that minimize total cost while ensuring
minimum and maximum nutritional intake across all nutrients.
"""

import pandas as pd

from cvxpy_or import (
    Model,
    Set,
    parameter_from_series,
    print_variable,
    variable_to_dataframe,
)

# =============================================================================
# INPUT DATA (as pandas DataFrames)
# =============================================================================

# Nutritional content per serving (food × nutrient matrix)
nutrition_df = pd.DataFrame(
    [
        {"food": "Bread", "nutrient": "Calories", "value": 80},
        {"food": "Bread", "nutrient": "Protein", "value": 3},
        {"food": "Bread", "nutrient": "Calcium", "value": 20},
        {"food": "Bread", "nutrient": "Fat", "value": 1},
        {"food": "Bread", "nutrient": "Carbs", "value": 15},
        {"food": "Milk", "nutrient": "Calories", "value": 150},
        {"food": "Milk", "nutrient": "Protein", "value": 8},
        {"food": "Milk", "nutrient": "Calcium", "value": 300},
        {"food": "Milk", "nutrient": "Fat", "value": 8},
        {"food": "Milk", "nutrient": "Carbs", "value": 12},
        {"food": "Cheese", "nutrient": "Calories", "value": 110},
        {"food": "Cheese", "nutrient": "Protein", "value": 7},
        {"food": "Cheese", "nutrient": "Calcium", "value": 200},
        {"food": "Cheese", "nutrient": "Fat", "value": 9},
        {"food": "Cheese", "nutrient": "Carbs", "value": 1},
        {"food": "Potato", "nutrient": "Calories", "value": 160},
        {"food": "Potato", "nutrient": "Protein", "value": 4},
        {"food": "Potato", "nutrient": "Calcium", "value": 20},
        {"food": "Potato", "nutrient": "Fat", "value": 0},
        {"food": "Potato", "nutrient": "Carbs", "value": 36},
        {"food": "Fish", "nutrient": "Calories", "value": 180},
        {"food": "Fish", "nutrient": "Protein", "value": 25},
        {"food": "Fish", "nutrient": "Calcium", "value": 30},
        {"food": "Fish", "nutrient": "Fat", "value": 8},
        {"food": "Fish", "nutrient": "Carbs", "value": 0},
        {"food": "Yogurt", "nutrient": "Calories", "value": 100},
        {"food": "Yogurt", "nutrient": "Protein", "value": 5},
        {"food": "Yogurt", "nutrient": "Calcium", "value": 150},
        {"food": "Yogurt", "nutrient": "Fat", "value": 2},
        {"food": "Yogurt", "nutrient": "Carbs", "value": 17},
    ]
)

# Cost per serving (dollars)
cost_series = pd.Series(
    {
        "Bread": 2.0,
        "Milk": 3.5,
        "Cheese": 8.0,
        "Potato": 1.5,
        "Fish": 11.0,
        "Yogurt": 1.0,
    },
    name="cost",
)

# Nutritional requirements (min and max per day)
requirements_df = pd.DataFrame(
    [
        {"nutrient": "Calories", "min": 2000, "max": 2500},
        {"nutrient": "Protein", "min": 50, "max": 200},
        {"nutrient": "Calcium", "min": 800, "max": 2000},
        {"nutrient": "Fat", "min": 0, "max": 65},
        {"nutrient": "Carbs", "min": 200, "max": 350},
    ]
)

print("=== Input Data ===")
print("\nFood Costs ($/serving):")
print(cost_series.to_frame().T)
print("\nNutritional Content (per serving):")
print(nutrition_df.pivot(index="food", columns="nutrient", values="value"))
print("\nDaily Requirements:")
print(requirements_df.set_index("nutrient"))
print()

# =============================================================================
# CREATE MODEL
# =============================================================================

m = Model(name="diet")

# Define sets from DataFrame
foods = Set(cost_series.index.tolist(), name="foods")
nutrients = Set(requirements_df["nutrient"].tolist(), name="nutrients")

print(f"Foods: {len(foods)}, Nutrients: {len(nutrients)}")
print()

# =============================================================================
# PARAMETERS
# =============================================================================

cost = parameter_from_series(cost_series, name="cost")

# Build nutrition matrix (nutrients x foods) from pivot table
nutrition_pivot = nutrition_df.pivot(index="nutrient", columns="food", values="value")
# Reorder to match our sets
nutrition_pivot = nutrition_pivot.reindex(index=list(nutrients), columns=list(foods))
nutrition_matrix = nutrition_pivot.values

# Requirements as series
min_req = parameter_from_series(requirements_df.set_index("nutrient")["min"], name="min_req")
max_req = parameter_from_series(requirements_df.set_index("nutrient")["max"], name="max_req")

# =============================================================================
# DECISION VARIABLES
# =============================================================================

buy = m.add_variable(foods, nonneg=True, name="buy")

# =============================================================================
# CONSTRAINTS
# =============================================================================

# Nutrient intake = nutrition_matrix @ buy (nutrients x foods) @ (foods,) = (nutrients,)
nutrient_intake = nutrition_matrix @ buy

# Minimum requirements
m.add_constraint("min_nutrients", nutrient_intake >= min_req)

# Maximum allowances
m.add_constraint("max_nutrients", nutrient_intake <= max_req)

# =============================================================================
# OBJECTIVE
# =============================================================================

m.minimize(cost @ buy)

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

# Export solution to DataFrame
print("=== Optimal Diet (as DataFrame) ===")
diet_df = variable_to_dataframe(buy, value_col="servings")
diet_df["servings"] = diet_df["servings"].round(2)
diet_df["cost_per_serving"] = diet_df["foods"].map(cost_series)
diet_df["total_cost"] = (diet_df["servings"] * diet_df["cost_per_serving"]).round(2)
diet_df = diet_df[diet_df["servings"] > 0.01]  # Non-zero only
print(diet_df.to_string(index=False))
print(f"\nTotal daily cost: ${diet_df['total_cost'].sum():.2f}")
print()

# Show nutritional content achieved
print("=== Nutritional Analysis ===")
nutrient_values = nutrient_intake.value
analysis_df = pd.DataFrame(
    {
        "nutrient": list(nutrients),
        "intake": [round(v, 1) for v in nutrient_values],
        "min_req": [min_req.get_value(n) for n in nutrients],
        "max_req": [max_req.get_value(n) for n in nutrients],
    }
)
analysis_df["status"] = analysis_df.apply(
    lambda row: "OK" if row["min_req"] <= row["intake"] <= row["max_req"] else "VIOLATION", axis=1
)
print(analysis_df.to_string(index=False))
print()

# Rich table display
print("=== Diet Plan (Rich Table) ===")
print_variable(buy, show_zero=False, precision=2)
