#!/usr/bin/env python3
"""Blending Problem using cvxpy-or.

This example demonstrates the classic blending/mixing problem:
mix raw materials to create a product that meets quality specifications
at minimum cost.

Problem: A feed company wants to blend ingredients to create animal feed
that meets nutritional requirements at minimum cost.
"""

import numpy as np

import cvxpy as cp

from cvxpy_or import Parameter, Set, Variable

# =============================================================================
# INDEX SETS
# =============================================================================

# Available raw materials (ingredients)
ingredients = Set(
    ['Corn', 'Oats', 'Soybean_Meal', 'Fish_Meal', 'Limestone'],
    name='ingredients'
)

# Quality attributes (properties) to control
properties = Set(
    ['Protein', 'Fat', 'Fiber', 'Calcium'],
    name='properties'
)

print(f"Ingredients: {len(ingredients)}")
print(f"Properties: {len(properties)}")
print()

# =============================================================================
# PARAMETERS
# =============================================================================

# Cost per kg of each ingredient ($/kg)
cost_data = {
    'Corn': 0.30,
    'Oats': 0.25,
    'Soybean_Meal': 0.45,
    'Fish_Meal': 0.80,
    'Limestone': 0.05,
}
cost = Parameter(ingredients, data=cost_data, name='cost')

# Composition: percentage of each property in each ingredient
# composition[i, p] = percentage of property p in ingredient i
composition_data = {
    #              Protein  Fat   Fiber  Calcium
    'Corn':        [8.0,    3.5,  2.0,   0.02],
    'Oats':        [11.0,   4.5,  10.0,  0.05],
    'Soybean_Meal': [44.0,   1.0,  7.0,   0.30],
    'Fish_Meal':   [60.0,   9.0,  0.5,   5.00],
    'Limestone':   [0.0,    0.0,  0.0,   38.0],
}

# Build composition matrix (properties x ingredients)
composition_matrix = np.array([
    [composition_data[i][j] for i in ingredients]
    for j, _ in enumerate(properties)
])

# Minimum specification (% in final blend)
min_spec_data = {
    'Protein': 20.0,   # At least 20% protein
    'Fat': 3.0,        # At least 3% fat
    'Fiber': 0.0,      # No minimum fiber
    'Calcium': 1.0,    # At least 1% calcium
}
min_spec = Parameter(properties, data=min_spec_data, name='min_spec')

# Maximum specification (% in final blend)
max_spec_data = {
    'Protein': 30.0,   # At most 30% protein
    'Fat': 8.0,        # At most 8% fat
    'Fiber': 8.0,      # At most 8% fiber
    'Calcium': 2.5,    # At most 2.5% calcium
}
max_spec = Parameter(properties, data=max_spec_data, name='max_spec')

# Total blend amount to produce (kg)
TOTAL_BLEND = 1000.0  # 1 ton

# =============================================================================
# DECISION VARIABLES
# =============================================================================

# Amount of each ingredient to use (kg)
blend = Variable(ingredients, nonneg=True, name='blend')

# =============================================================================
# OBJECTIVE: Minimize total ingredient cost
# =============================================================================

objective = cp.Minimize(cost @ blend)

# =============================================================================
# CONSTRAINTS
# =============================================================================

# 1. Total blend must equal target quantity
#    sum of all ingredients = TOTAL_BLEND

# 2. Property bounds: percentage in blend must be within specs
#    For each property p:
#    min_spec[p] * TOTAL_BLEND / 100 <= sum_i composition[i,p] * blend[i] <= max_spec[p] * TOTAL_BLEND / 100
#
#    Equivalently (as percentage of total blend):
#    min_spec[p] <= (sum_i composition[i,p] * blend[i]) / TOTAL_BLEND * 100 <= max_spec[p]

# Property content (in kg) = composition_matrix @ blend
# Property percentage = property_content / TOTAL_BLEND * 100
# But since composition is already in %, we get: property_content = composition_matrix @ blend / 100 * blend_amount
# Actually: composition_matrix @ blend gives weighted % contribution
# To get % in final blend: (composition_matrix @ blend) / TOTAL_BLEND if composition is fraction

# composition[i,p] is % of property p in ingredient i
# So: sum_i (composition[i,p] / 100) * blend[i] = kg of property p
# Property % in blend = (kg of property p) / TOTAL_BLEND * 100
#                     = sum_i composition[i,p] * blend[i] / TOTAL_BLEND

# Compute property percentages in final blend
property_percent = (composition_matrix @ blend) / TOTAL_BLEND

constraints = [
    # Total blend equals target
    cp.sum(blend) == TOTAL_BLEND,

    # Property lower bounds (as percentage)
    property_percent >= min_spec,

    # Property upper bounds (as percentage)
    property_percent <= max_spec,
]

# =============================================================================
# SOLVE
# =============================================================================

prob = cp.Problem(objective, constraints)
result = prob.solve()

print(f"Status: {prob.status}")
print(f"Total blend cost: ${result:.2f}")
print(f"Cost per kg: ${result / TOTAL_BLEND:.4f}")
print()

# =============================================================================
# RESULTS
# =============================================================================

print("=== Optimal Blend Recipe ===")
for i in ingredients:
    val = blend.get_value(i)
    if val > 0.01:
        pct = val / TOTAL_BLEND * 100
        ingredient_cost = cost.get_value(i) * val
        print(f"  {i:14}: {val:7.1f} kg ({pct:5.1f}%) - ${ingredient_cost:.2f}")

print()
print(f"Total: {TOTAL_BLEND:.0f} kg")

print()
print("=== Final Blend Composition ===")
property_values = property_percent.value
for j, p in enumerate(properties):
    print(f"  {p:10}: {property_values[j]:5.1f}% (min: {min_spec.get_value(p)}%, max: {max_spec.get_value(p)}%)")
