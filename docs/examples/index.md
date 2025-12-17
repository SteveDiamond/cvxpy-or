# Examples

These examples demonstrate cvxpy-or on classic Operations Research problems.

```{toctree}
:maxdepth: 1

transportation
assignment
facility-location
diet
blending
```

## Overview

| Example | Problem Type | Key Features |
|---------|-------------|--------------|
| [Transportation](transportation.md) | Network flow | Multi-period, inventory, `sum_by` |
| [Assignment](assignment.md) | Assignment | Binary constraints, pandas I/O |
| [Facility Location](facility-location.md) | Facility location | Linking constraints, validation |
| [Diet](diet.md) | Resource allocation | Matrix constraints, nutrient bounds |
| [Blending](blending.md) | Blending | Percentage constraints, composition |

## Running Examples

The example files are in the `examples/` directory:

```bash
# Run the transportation example
python examples/multi_period_transportation.py

# Run all examples
for f in examples/*.py; do python "$f"; done
```

## What You'll Learn

### Transportation Problem
- Multi-dimensional indexing (warehouse × customer × period)
- Using `sum_by` for supply and demand constraints
- Inventory tracking across periods

### Assignment Problem
- Creating binary assignment variables
- One-to-one matching constraints
- Exporting solutions to DataFrames

### Facility Location
- Linking constraints between open/closed and flow variables
- Data validation with helpful error messages
- Cost breakdown analysis

### Diet Problem
- Matrix-based nutritional constraints
- Min/max bounds on aggregated values
- Nutritional analysis of the solution

### Blending Problem
- Percentage-based composition constraints
- Total quantity constraints
- Cost composition analysis
