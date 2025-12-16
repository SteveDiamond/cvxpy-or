#!/usr/bin/env python3
"""Blending Problem using cvxpy-or.

This example demonstrates:
- Loading composition data from pandas DataFrames
- Percentage-based constraints
- Detailed cost breakdown analysis
- Rich table visualization

Problem: A feed company wants to blend ingredients to create animal feed
that meets nutritional requirements at minimum cost.
"""

import pandas as pd

import cvxpy as cp

from cvxpy_or import (
    Model,
    Set,
    parameter_from_series,
    variable_to_dataframe,
    print_variable,
)

# =============================================================================
# INPUT DATA (as pandas DataFrames)
# =============================================================================

# Ingredient composition (% of each property in each ingredient)
composition_df = pd.DataFrame([
    {'ingredient': 'Corn', 'property': 'Protein', 'pct': 8.0},
    {'ingredient': 'Corn', 'property': 'Fat', 'pct': 3.5},
    {'ingredient': 'Corn', 'property': 'Fiber', 'pct': 2.0},
    {'ingredient': 'Corn', 'property': 'Calcium', 'pct': 0.02},
    {'ingredient': 'Oats', 'property': 'Protein', 'pct': 11.0},
    {'ingredient': 'Oats', 'property': 'Fat', 'pct': 4.5},
    {'ingredient': 'Oats', 'property': 'Fiber', 'pct': 10.0},
    {'ingredient': 'Oats', 'property': 'Calcium', 'pct': 0.05},
    {'ingredient': 'Soybean_Meal', 'property': 'Protein', 'pct': 44.0},
    {'ingredient': 'Soybean_Meal', 'property': 'Fat', 'pct': 1.0},
    {'ingredient': 'Soybean_Meal', 'property': 'Fiber', 'pct': 7.0},
    {'ingredient': 'Soybean_Meal', 'property': 'Calcium', 'pct': 0.30},
    {'ingredient': 'Fish_Meal', 'property': 'Protein', 'pct': 60.0},
    {'ingredient': 'Fish_Meal', 'property': 'Fat', 'pct': 9.0},
    {'ingredient': 'Fish_Meal', 'property': 'Fiber', 'pct': 0.5},
    {'ingredient': 'Fish_Meal', 'property': 'Calcium', 'pct': 5.00},
    {'ingredient': 'Limestone', 'property': 'Protein', 'pct': 0.0},
    {'ingredient': 'Limestone', 'property': 'Fat', 'pct': 0.0},
    {'ingredient': 'Limestone', 'property': 'Fiber', 'pct': 0.0},
    {'ingredient': 'Limestone', 'property': 'Calcium', 'pct': 38.0},
])

# Cost per kg of each ingredient ($/kg)
cost_series = pd.Series({
    'Corn': 0.30,
    'Oats': 0.25,
    'Soybean_Meal': 0.45,
    'Fish_Meal': 0.80,
    'Limestone': 0.05,
}, name='cost')

# Property specifications (min and max % in final blend)
specs_df = pd.DataFrame([
    {'property': 'Protein', 'min_pct': 20.0, 'max_pct': 30.0},
    {'property': 'Fat', 'min_pct': 3.0, 'max_pct': 8.0},
    {'property': 'Fiber', 'min_pct': 0.0, 'max_pct': 8.0},
    {'property': 'Calcium', 'min_pct': 1.0, 'max_pct': 2.5},
])

# Total blend to produce
TOTAL_BLEND = 1000.0  # kg

print("=== Input Data ===")
print("\nIngredient Costs ($/kg):")
print(cost_series.to_frame().T)
print("\nIngredient Composition (%):")
print(composition_df.pivot(index='ingredient', columns='property', values='pct'))
print("\nBlend Specifications (%):")
print(specs_df.set_index('property'))
print(f"\nTotal blend to produce: {TOTAL_BLEND:.0f} kg")
print()

# =============================================================================
# CREATE MODEL
# =============================================================================

m = Model(name='blending')

# Define sets from DataFrame
ingredients = Set(cost_series.index.tolist(), name='ingredients')
properties = Set(specs_df['property'].tolist(), name='properties')

print(f"Ingredients: {len(ingredients)}, Properties: {len(properties)}")
print()

# =============================================================================
# PARAMETERS
# =============================================================================

cost = parameter_from_series(cost_series, name='cost')

# Build composition matrix (properties x ingredients) from pivot table
composition_pivot = composition_df.pivot(index='property', columns='ingredient', values='pct')
composition_pivot = composition_pivot.reindex(index=list(properties), columns=list(ingredients))
composition_matrix = composition_pivot.values

# Specifications as series
min_spec = parameter_from_series(
    specs_df.set_index('property')['min_pct'],
    name='min_spec'
)
max_spec = parameter_from_series(
    specs_df.set_index('property')['max_pct'],
    name='max_spec'
)

# =============================================================================
# DECISION VARIABLES
# =============================================================================

blend = m.add_variable(ingredients, nonneg=True, name='blend')

# =============================================================================
# CONSTRAINTS
# =============================================================================

# Property percentage in final blend = (composition_matrix @ blend) / TOTAL_BLEND
property_percent = (composition_matrix @ blend) / TOTAL_BLEND

# Total blend must equal target
m.add_constraint('total', cp.sum(blend) == TOTAL_BLEND)

# Property bounds
m.add_constraint('min_spec', property_percent >= min_spec)
m.add_constraint('max_spec', property_percent <= max_spec)

# =============================================================================
# OBJECTIVE
# =============================================================================

m.minimize(cost @ blend)

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

total_cost = m.value
print(f"Total blend cost: ${total_cost:.2f}")
print(f"Cost per kg: ${total_cost / TOTAL_BLEND:.4f}")
print()

# Export solution to DataFrame
print("=== Optimal Blend Recipe (as DataFrame) ===")
recipe_df = variable_to_dataframe(blend, value_col='kg')
recipe_df['kg'] = recipe_df['kg'].round(1)
recipe_df['pct'] = (recipe_df['kg'] / TOTAL_BLEND * 100).round(1)
recipe_df['cost_per_kg'] = recipe_df['ingredients'].map(cost_series)
recipe_df['total_cost'] = (recipe_df['kg'] * recipe_df['cost_per_kg']).round(2)
recipe_df = recipe_df[recipe_df['kg'] > 0.01]  # Non-zero only
print(recipe_df.to_string(index=False))
print(f"\nTotal: {recipe_df['kg'].sum():.0f} kg, Cost: ${recipe_df['total_cost'].sum():.2f}")
print()

# Show final blend composition
print("=== Final Blend Composition ===")
property_values = property_percent.value
composition_analysis = pd.DataFrame({
    'property': list(properties),
    'actual_pct': [round(v, 2) for v in property_values],
    'min_pct': [min_spec.get_value(p) for p in properties],
    'max_pct': [max_spec.get_value(p) for p in properties],
})
composition_analysis['status'] = composition_analysis.apply(
    lambda row: 'OK' if row['min_pct'] <= row['actual_pct'] <= row['max_pct'] else 'VIOLATION',
    axis=1
)
print(composition_analysis.to_string(index=False))
print()

# Cost breakdown by ingredient (pie chart style)
print("=== Cost Breakdown ===")
cost_breakdown = recipe_df[['ingredients', 'total_cost']].copy()
cost_breakdown['share_pct'] = (cost_breakdown['total_cost'] / cost_breakdown['total_cost'].sum() * 100).round(1)
print(cost_breakdown.to_string(index=False))
print()

# Rich table display
print("=== Blend Recipe (Rich Table) ===")
print_variable(blend, show_zero=False, precision=1)
