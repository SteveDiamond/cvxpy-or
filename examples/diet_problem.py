#!/usr/bin/env python3
"""Diet Problem using cvxpy-or.

This example demonstrates the classic diet optimization problem:
minimize the cost of a diet while meeting nutritional requirements.

Problem: Select foods to purchase that minimize total cost while ensuring
minimum and maximum nutritional intake across all nutrients.
"""

import numpy as np

import cvxpy as cp

from cvxpy_or import Parameter, Set, Variable

# =============================================================================
# INDEX SETS
# =============================================================================

# Available foods
foods = Set(
    ['Bread', 'Milk', 'Cheese', 'Potato', 'Fish', 'Yogurt'],
    name='foods'
)

# Nutrients to track
nutrients = Set(
    ['Calories', 'Protein', 'Calcium', 'Fat', 'Carbs'],
    name='nutrients'
)

print(f"Foods: {len(foods)}")
print(f"Nutrients: {len(nutrients)}")
print()

# =============================================================================
# PARAMETERS
# =============================================================================

# Cost per unit of each food (dollars per serving)
cost_data = {
    'Bread': 2.0,
    'Milk': 3.5,
    'Cheese': 8.0,
    'Potato': 1.5,
    'Fish': 11.0,
    'Yogurt': 1.0,
}
cost = Parameter(foods, data=cost_data, name='cost')

# Nutritional content per serving (simplified values)
# Matrix: rows = nutrients, cols = foods
# This allows: nutrition_matrix @ buy = nutrient totals
nutrition_data = {
    #           Bread  Milk  Cheese  Potato  Fish  Yogurt
    'Calories': [80,   150,  110,    160,    180,  100],
    'Protein':  [3,    8,    7,      4,      25,   5],
    'Calcium':  [20,   300,  200,    20,     30,   150],
    'Fat':      [1,    8,    9,      0,      8,    2],
    'Carbs':    [15,   12,   1,      36,     0,    17],
}

# Build nutrition matrix (nutrients x foods)
nutrition_matrix = np.array([nutrition_data[n] for n in nutrients])

# Minimum daily requirements
min_req_data = {
    'Calories': 2000,
    'Protein': 50,
    'Calcium': 800,
    'Fat': 0,      # No minimum for fat
    'Carbs': 200,
}
min_req = Parameter(nutrients, data=min_req_data, name='min_req')

# Maximum daily allowances
max_req_data = {
    'Calories': 2500,
    'Protein': 200,   # High limit
    'Calcium': 2000,
    'Fat': 65,
    'Carbs': 350,
}
max_req = Parameter(nutrients, data=max_req_data, name='max_req')

# =============================================================================
# DECISION VARIABLES
# =============================================================================

# Amount of each food to purchase (servings per day)
buy = Variable(foods, nonneg=True, name='buy')

# =============================================================================
# OBJECTIVE: Minimize total food cost
# =============================================================================

objective = cp.Minimize(cost @ buy)

# =============================================================================
# CONSTRAINTS
# =============================================================================

# For each nutrient, we need:
#   min_req[n] <= sum over foods of nutrition[f,n] * buy[f] <= max_req[n]
#
# Using matrix multiplication: nutrition_matrix @ buy gives nutrient totals
# nutrition_matrix is (nutrients x foods), buy is (foods,)
# Result is (nutrients,) vector of total nutrient intake

nutrient_intake = nutrition_matrix @ buy

constraints = [
    # Minimum requirements
    nutrient_intake >= min_req,

    # Maximum allowances
    nutrient_intake <= max_req,
]

# =============================================================================
# SOLVE
# =============================================================================

prob = cp.Problem(objective, constraints)
result = prob.solve()

print(f"Status: {prob.status}")
print(f"Optimal daily cost: ${result:.2f}")
print()

# =============================================================================
# RESULTS
# =============================================================================

print("=== Optimal Diet (servings per day) ===")
for f in foods:
    val = buy.get_value(f)
    if val > 0.01:
        print(f"  {f:10}: {val:5.2f} servings (${cost.get_value(f) * val:.2f})")

print()
print("=== Nutritional Content ===")
nutrient_values = nutrient_intake.value
for i, n in enumerate(nutrients):
    print(f"  {n:10}: {nutrient_values[i]:7.1f} (min: {min_req.get_value(n)}, max: {max_req.get_value(n)})")
