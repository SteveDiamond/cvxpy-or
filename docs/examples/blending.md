# Blending Problem

This example solves a blending problem: mix ingredients to create a product meeting specifications at minimum cost.

**Source**: `examples/blending_problem.py`

## Problem Description

A feed company blends ingredients (corn, oats, soybean meal, fish meal, limestone) to create animal feed. Each ingredient has a cost and contributes to the feed's nutritional properties. The final blend must meet minimum and maximum specifications for protein, fat, fiber, and calcium content.

**Objective**: Minimize total ingredient cost while meeting all specifications.

## Mathematical Formulation

**Sets**:
- $I$ = ingredients
- $P$ = properties (nutrients)

**Parameters**:
- $cost_i$ = cost per kg of ingredient $i$
- $composition_{p,i}$ = percentage of property $p$ in ingredient $i$
- $min\_spec_p$ = minimum percentage of property $p$ in final blend
- $max\_spec_p$ = maximum percentage of property $p$ in final blend
- $TOTAL$ = total kg of blend to produce

**Variables**:
- $blend_i \geq 0$ = kg of ingredient $i$ to use

**Constraints**:
- Total blend: $\sum_i blend_i = TOTAL$
- Min spec: $\frac{\sum_i composition_{p,i} \cdot blend_i}{TOTAL} \geq min\_spec_p$ for all $p$
- Max spec: $\frac{\sum_i composition_{p,i} \cdot blend_i}{TOTAL} \leq max\_spec_p$ for all $p$

**Objective**: Minimize $\sum_i cost_i \cdot blend_i$

## Implementation

### Data Setup

```python
import pandas as pd
import cvxpy as cp
from cvxpy_or import (
    Model, Set,
    parameter_from_series, variable_to_dataframe, print_variable,
)

# Ingredient composition (% of each property)
composition_df = pd.DataFrame([
    {'ingredient': 'Corn', 'property': 'Protein', 'pct': 8.0},
    {'ingredient': 'Corn', 'property': 'Fat', 'pct': 3.5},
    {'ingredient': 'Corn', 'property': 'Fiber', 'pct': 2.0},
    {'ingredient': 'Soybean_Meal', 'property': 'Protein', 'pct': 44.0},
    {'ingredient': 'Fish_Meal', 'property': 'Protein', 'pct': 60.0},
    # ... more composition data
])

# Cost per kg
cost_series = pd.Series({
    'Corn': 0.30,
    'Oats': 0.25,
    'Soybean_Meal': 0.45,
    'Fish_Meal': 0.80,
    'Limestone': 0.05,
}, name='cost')

# Specifications (min and max %)
specs_df = pd.DataFrame([
    {'property': 'Protein', 'min_pct': 20.0, 'max_pct': 30.0},
    {'property': 'Fat', 'min_pct': 3.0, 'max_pct': 8.0},
    {'property': 'Fiber', 'min_pct': 0.0, 'max_pct': 8.0},
    {'property': 'Calcium', 'min_pct': 1.0, 'max_pct': 2.5},
])

TOTAL_BLEND = 1000.0  # kg
```

### Model Definition

```python
m = Model(name='blending')

# Define sets
ingredients = Set(cost_series.index.tolist(), name='ingredients')
properties = Set(specs_df['property'].tolist(), name='properties')
```

### Parameters

```python
cost = parameter_from_series(cost_series, name='cost')

# Build composition matrix (properties x ingredients)
composition_pivot = composition_df.pivot(
    index='property', columns='ingredient', values='pct')
composition_pivot = composition_pivot.reindex(
    index=list(properties), columns=list(ingredients))
composition_matrix = composition_pivot.values

# Specifications
min_spec = parameter_from_series(
    specs_df.set_index('property')['min_pct'], name='min_spec')
max_spec = parameter_from_series(
    specs_df.set_index('property')['max_pct'], name='max_spec')
```

### Variables and Constraints

```python
# Decision variable: kg of each ingredient
blend = m.add_variable(ingredients, nonneg=True, name='blend')

# Property percentage in final blend
property_percent = (composition_matrix @ blend) / TOTAL_BLEND

# Total blend must equal target
m.add_constraint('total', cp.sum(blend) == TOTAL_BLEND)

# Property bounds
m.add_constraint('min_spec', property_percent >= min_spec)
m.add_constraint('max_spec', property_percent <= max_spec)

# Objective
m.minimize(cost @ blend)
```

### Solve and Analyze

```python
m.solve()

total_cost = m.value
print(f"Total cost: ${total_cost:.2f}")
print(f"Cost per kg: ${total_cost / TOTAL_BLEND:.4f}")

# Export recipe
recipe_df = variable_to_dataframe(blend, value_col='kg')
recipe_df['pct'] = recipe_df['kg'] / TOTAL_BLEND * 100
recipe_df['cost_per_kg'] = recipe_df['ingredients'].map(cost_series)
recipe_df['total_cost'] = recipe_df['kg'] * recipe_df['cost_per_kg']
recipe_df = recipe_df[recipe_df['kg'] > 0.01]
print(recipe_df)

# Composition analysis
property_values = property_percent.value
analysis = pd.DataFrame({
    'property': list(properties),
    'actual_pct': property_values,
    'min_pct': [min_spec.get_value(p) for p in properties],
    'max_pct': [max_spec.get_value(p) for p in properties],
})
print(analysis)
```

## Key Patterns

### Percentage Constraints

The key insight is that percentages in the final blend depend on the ratio:

```python
# Property content in blend
property_content = composition_matrix @ blend  # kg of each property

# Convert to percentage
property_percent = property_content / TOTAL_BLEND

# Compare to specification percentages
m.add_constraint('min_spec', property_percent >= min_spec)
m.add_constraint('max_spec', property_percent <= max_spec)
```

### Total Quantity Constraint

```python
# The blend must sum to the target quantity
m.add_constraint('total', cp.sum(blend) == TOTAL_BLEND)
```

### Cost Analysis

```python
# Cost per unit of output
unit_cost = m.value / TOTAL_BLEND

# Cost breakdown by ingredient
recipe_df['share_pct'] = recipe_df['total_cost'] / m.value * 100
```

## Variations

### Minimum/Maximum Ingredient Usage

```python
# At least 10% corn
m.add_constraint('min_corn', blend['Corn'] >= 0.10 * TOTAL_BLEND)

# At most 5% fish meal
m.add_constraint('max_fish', blend['Fish_Meal'] <= 0.05 * TOTAL_BLEND)
```

### Ingredient Availability

```python
available = m.add_parameter(ingredients, data={
    'Corn': 500, 'Oats': 300, 'Soybean_Meal': 200, ...})
m.add_constraint('availability', blend <= available)
```

### Multiple Products

```python
products = Set(['Feed_A', 'Feed_B'], name='products')
blend = m.add_variable(Set.cross(ingredients, products), nonneg=True)

# Each product has its own specifications
for product in products:
    product_blend = where(blend, products=product)
    # Add product-specific constraints
```

### Ratio Constraints

```python
# Corn-to-oats ratio between 2:1 and 4:1
m.add_constraint('ratio_min', blend['Corn'] >= 2 * blend['Oats'])
m.add_constraint('ratio_max', blend['Corn'] <= 4 * blend['Oats'])
```

### Variable Batch Size

```python
# Remove fixed total, add minimum production
min_production = 500
m.add_constraint('min_total', cp.sum(blend) >= min_production)

# Objective includes per-unit cost anyway
m.minimize(cost @ blend)
```
