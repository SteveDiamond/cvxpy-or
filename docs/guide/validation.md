# Validation

cvxpy-or provides validation utilities that give helpful error messages when data doesn't match expected formats.

## ValidationError

All validation functions raise `ValidationError` with descriptive messages:

```python
from cvxpy_or import ValidationError

try:
    # Some operation that might fail
    param = m.add_parameter(items, data=bad_data)
except ValidationError as e:
    print(f"Data error: {e}")
```

## validate_keys

Check that data keys match an index, with suggestions for typos:

```python
from cvxpy_or import validate_keys, Set

items = Set(['Widget', 'Gadget', 'Gizmo'], name='items')

data = {
    'Widget': 10,
    'Gadjet': 20,  # Typo!
    'Gizmo': 15,
}

validate_keys(data, items)
# Raises ValidationError:
# "Key 'Gadjet' not found in index. Did you mean 'Gadget'?"
```

### Partial Validation

Allow missing keys:

```python
validate_keys(data, items, allow_missing=True)
```

Allow extra keys:

```python
validate_keys(data, items, allow_extra=True)
```

## validate_numeric

Ensure all values are numeric:

```python
from cvxpy_or import validate_numeric

data = {'A': 10, 'B': 'twenty', 'C': 30}

validate_numeric(data)
# Raises ValidationError:
# "Value for key 'B' is not numeric: 'twenty'"
```

### NaN and Inf

By default, NaN and Inf are not allowed:

```python
import numpy as np

data = {'A': 10, 'B': np.nan}
validate_numeric(data)
# Raises ValidationError

# Allow NaN
validate_numeric(data, allow_nan=True)

# Allow Inf
data = {'A': 10, 'B': np.inf}
validate_numeric(data, allow_inf=True)
```

## validate_bounds

Check that values are within bounds:

```python
from cvxpy_or import validate_bounds

data = {'A': 10, 'B': -5, 'C': 30}

validate_bounds(data, lower=0)
# Raises ValidationError:
# "Value for key 'B' is -5, which is below lower bound 0"

validate_bounds(data, upper=20)
# Raises ValidationError:
# "Value for key 'C' is 30, which is above upper bound 20"

validate_bounds(data, lower=0, upper=100)
# Both bounds at once
```

## validate_parameter

Comprehensive validation for parameter data:

```python
from cvxpy_or import validate_parameter

# Validates keys, numeric types, and optional bounds
validate_parameter(
    data,
    index=items,
    lower=0,
    upper=100,
    name='cost'  # Used in error messages
)
```

## Automatic Validation

When you create Parameters through the Model, validation is automatic:

```python
m = Model()
items = Set(['A', 'B', 'C'])

# This will validate automatically
cost = m.add_parameter(items, data={'A': 10, 'B': 20, 'C': 30})

# Invalid data raises ValidationError
cost = m.add_parameter(items, data={'A': 10, 'X': 20})
# ValidationError: Key 'X' not found in index...
```

## Typo Detection

The validation system uses fuzzy matching to suggest corrections:

```python
cities = Set(['New York', 'Los Angeles', 'Chicago', 'Houston'])

data = {
    'New York': 100,
    'Los Angles': 80,  # Typo
    'Chicago': 90,
    'Huston': 70,      # Typo
}

validate_keys(data, cities)
# ValidationError: Keys not found in index:
#   'Los Angles' - did you mean 'Los Angeles'?
#   'Huston' - did you mean 'Houston'?
```

## Debugging Tips

### Check Set Contents

```python
print(f"Index contains: {list(items)}")
print(f"Data keys: {list(data.keys())}")
```

### Find Mismatches

```python
index_set = set(items)
data_set = set(data.keys())

missing = index_set - data_set
extra = data_set - index_set

print(f"Missing from data: {missing}")
print(f"Extra in data: {extra}")
```

### Validate Incrementally

```python
# Validate keys first
validate_keys(data, items)

# Then validate values
validate_numeric(data)

# Then validate bounds
validate_bounds(data, lower=0)
```

## Best Practices

1. **Validate early**: Check data when loading, not when solving
2. **Use descriptive names**: Include `name` parameter for clear error messages
3. **Handle errors gracefully**: Catch `ValidationError` and provide user feedback
4. **Test with edge cases**: Empty data, single elements, large datasets

```python
def load_cost_data(filepath, items):
    """Load and validate cost data from CSV."""
    df = pd.read_csv(filepath)
    data = dict(zip(df['item'], df['cost']))

    try:
        validate_keys(data, items)
        validate_numeric(data)
        validate_bounds(data, lower=0, name='cost')
    except ValidationError as e:
        raise ValueError(f"Invalid cost data in {filepath}: {e}")

    return data
```
