# Diet Problem

This example solves the classic diet problem: select foods to minimize cost while meeting nutritional requirements.

**Source**: `examples/diet_problem.py`

## Problem Description

A person wants to plan their daily diet. Each food has a cost per serving and provides various nutrients. There are minimum and maximum requirements for each nutrient.

**Objective**: Minimize total food cost while meeting all nutritional requirements.

## Mathematical Formulation

**Sets**:
- $F$ = foods
- $N$ = nutrients

**Parameters**:
- $cost_f$ = cost per serving of food $f$
- $nutrition_{n,f}$ = amount of nutrient $n$ in one serving of food $f$
- $min_n$ = minimum daily requirement for nutrient $n$
- $max_n$ = maximum daily allowance for nutrient $n$

**Variables**:
- $buy_f \geq 0$ = servings of food $f$ to buy

**Constraints**:
- Minimum nutrients: $\sum_f nutrition_{n,f} \cdot buy_f \geq min_n$ for all $n$
- Maximum nutrients: $\sum_f nutrition_{n,f} \cdot buy_f \leq max_n$ for all $n$

**Objective**: Minimize $\sum_f cost_f \cdot buy_f$

## Implementation

### Data Setup

```python
import pandas as pd
import cvxpy as cp
from cvxpy_or import (
    Model, Set,
    parameter_from_series, variable_to_dataframe, print_variable,
)

# Nutritional content per serving
nutrition_df = pd.DataFrame([
    {'food': 'Bread', 'nutrient': 'Calories', 'value': 80},
    {'food': 'Bread', 'nutrient': 'Protein', 'value': 3},
    {'food': 'Bread', 'nutrient': 'Calcium', 'value': 20},
    {'food': 'Milk', 'nutrient': 'Calories', 'value': 150},
    {'food': 'Milk', 'nutrient': 'Protein', 'value': 8},
    # ... more nutrition data
])

# Cost per serving
cost_series = pd.Series({
    'Bread': 2.0,
    'Milk': 3.5,
    'Cheese': 8.0,
    'Potato': 1.5,
    'Fish': 11.0,
    'Yogurt': 1.0,
}, name='cost')

# Nutritional requirements
requirements_df = pd.DataFrame([
    {'nutrient': 'Calories', 'min': 2000, 'max': 2500},
    {'nutrient': 'Protein', 'min': 50, 'max': 200},
    {'nutrient': 'Calcium', 'min': 800, 'max': 2000},
    {'nutrient': 'Fat', 'min': 0, 'max': 65},
    {'nutrient': 'Carbs', 'min': 200, 'max': 350},
])
```

### Model Definition

```python
m = Model(name='diet')

# Define sets
foods = Set(cost_series.index.tolist(), name='foods')
nutrients = Set(requirements_df['nutrient'].tolist(), name='nutrients')
```

### Parameters

```python
cost = parameter_from_series(cost_series, name='cost')

# Build nutrition matrix (nutrients x foods) from pivot table
nutrition_pivot = nutrition_df.pivot(
    index='nutrient', columns='food', values='value')
nutrition_pivot = nutrition_pivot.reindex(
    index=list(nutrients), columns=list(foods))
nutrition_matrix = nutrition_pivot.values

# Requirements
min_req = parameter_from_series(
    requirements_df.set_index('nutrient')['min'], name='min_req')
max_req = parameter_from_series(
    requirements_df.set_index('nutrient')['max'], name='max_req')
```

### Variables and Constraints

```python
# Decision variable: servings to buy
buy = m.add_variable(foods, nonneg=True, name='buy')

# Nutrient intake = nutrition_matrix @ buy
nutrient_intake = nutrition_matrix @ buy

# Minimum requirements
m.add_constraint('min_nutrients', nutrient_intake >= min_req)

# Maximum allowances
m.add_constraint('max_nutrients', nutrient_intake <= max_req)

# Objective
m.minimize(cost @ buy)
```

### Solve and Analyze

```python
m.solve()
m.print_summary()

# Export solution
diet_df = variable_to_dataframe(buy, value_col='servings')
diet_df['cost'] = diet_df['foods'].map(cost_series)
diet_df['total_cost'] = diet_df['servings'] * diet_df['cost']
diet_df = diet_df[diet_df['servings'] > 0.01]
print(diet_df)

# Nutritional analysis
nutrient_values = nutrient_intake.value
analysis = pd.DataFrame({
    'nutrient': list(nutrients),
    'intake': nutrient_values,
    'min': [min_req.get_value(n) for n in nutrients],
    'max': [max_req.get_value(n) for n in nutrients],
})
print(analysis)
```

## Key Patterns

### Matrix Constraints

The nutrition constraint is a matrix multiplication:

```python
# nutrition_matrix is (nutrients x foods)
# buy is (foods,)
# Result is (nutrients,)
nutrient_intake = nutrition_matrix @ buy

# Then compare to bounds
m.add_constraint('min', nutrient_intake >= min_req)
m.add_constraint('max', nutrient_intake <= max_req)
```

### Building Matrices from DataFrames

```python
# Pivot long-form data to matrix
nutrition_pivot = nutrition_df.pivot(
    index='nutrient', columns='food', values='value')

# Ensure rows/columns match Set order
nutrition_pivot = nutrition_pivot.reindex(
    index=list(nutrients), columns=list(foods))

# Convert to numpy array
nutrition_matrix = nutrition_pivot.values
```

### Nutritional Analysis

```python
# After solving, evaluate the constraint expressions
nutrient_values = nutrient_intake.value  # numpy array

# Check which constraints are binding
for i, nutrient in enumerate(nutrients):
    intake = nutrient_values[i]
    if intake <= min_req.get_value(nutrient) + 0.01:
        print(f"{nutrient}: at minimum")
    elif intake >= max_req.get_value(nutrient) - 0.01:
        print(f"{nutrient}: at maximum")
```

## Variations

### Integer Servings

```python
buy = m.add_variable(foods, integer=True, nonneg=True, name='buy')
```

### Maximum Servings per Food

```python
max_servings = m.add_parameter(foods, data={'Bread': 5, 'Milk': 4, ...})
m.add_constraint('max_serving', buy <= max_servings)
```

### Food Groups

Ensure variety by requiring foods from different groups:

```python
from cvxpy_or import at_least_k

# At least 3 different foods
buy_positive = m.add_variable(foods, boolean=True)
m.add_constraint('link', buy <= 1000 * buy_positive)  # Big-M
constraints = at_least_k(buy_positive, k=3)
m.add_constraint('variety', constraints)
```

### Meal Planning

Extend to multiple meals:

```python
meals = Set(['breakfast', 'lunch', 'dinner'], name='meals')
buy = m.add_variable(Set.cross(foods, meals), nonneg=True)

# Requirements per meal
m.add_constraint('meal_nutrients',
    nutrition_matrix @ sum_by(buy, 'meals') >= min_req)
```
