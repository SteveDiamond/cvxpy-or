"""Tests for cvxpy_or.validation module."""

import unittest

from cvxpy_or import (
    Parameter,
    Set,
    ValidationError,
    validate_bounds,
    validate_keys,
    validate_numeric,
    validate_parameter,
)


class TestValidateKeys(unittest.TestCase):
    """Tests for validate_keys function."""

    def test_valid_complete_data(self):
        """Test validation passes with complete valid data."""
        idx = Set(['a', 'b', 'c'], name='letters')
        data = {'a': 1, 'b': 2, 'c': 3}
        # Should not raise
        validate_keys(data, idx)

    def test_missing_keys(self):
        """Test error when keys are missing."""
        idx = Set(['a', 'b', 'c'], name='letters')
        data = {'a': 1, 'b': 2}  # missing 'c'
        with self.assertRaises(ValidationError) as ctx:
            validate_keys(data, idx)
        self.assertIn("Missing", str(ctx.exception))
        self.assertIn("1 key(s)", str(ctx.exception))

    def test_allow_partial(self):
        """Test partial data is allowed with flag."""
        idx = Set(['a', 'b', 'c'], name='letters')
        data = {'a': 1}
        # Should not raise
        validate_keys(data, idx, allow_partial=True)

    def test_extra_key_simple_index(self):
        """Test error when extra key in simple index."""
        idx = Set(['a', 'b', 'c'], name='letters')
        data = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
        with self.assertRaises(ValidationError) as ctx:
            validate_keys(data, idx)
        self.assertIn("Invalid key", str(ctx.exception))
        self.assertIn("'d'", str(ctx.exception))

    def test_extra_key_compound_index(self):
        """Test error with typo in compound index."""
        idx = Set(
            [('W1', 'C1'), ('W1', 'C2'), ('W2', 'C1'), ('W2', 'C2')],
            name='routes',
            names=('warehouse', 'customer'),
        )
        data = {
            ('W1', 'C1'): 10,
            ('W1', 'C2'): 15,
            ('W2', 'C1'): 20,
            ('TYPO', 'C2'): 25,  # typo!
        }
        with self.assertRaises(ValidationError) as ctx:
            validate_keys(data, idx, allow_partial=True)
        error_msg = str(ctx.exception)
        self.assertIn("Invalid key", error_msg)
        self.assertIn("TYPO", error_msg)
        self.assertIn("Position 0", error_msg)
        # Should suggest similar values
        self.assertIn("W1", error_msg)

    def test_wrong_tuple_length(self):
        """Test error when tuple has wrong length."""
        idx = Set([('W1', 'C1'), ('W1', 'C2')], name='routes')
        data = {('W1', 'C1'): 10, ('W1', 'C2', 'extra'): 15}
        with self.assertRaises(ValidationError) as ctx:
            validate_keys(data, idx, allow_partial=True)
        self.assertIn("Expected tuple of length 2", str(ctx.exception))


class TestValidateNumeric(unittest.TestCase):
    """Tests for validate_numeric function."""

    def test_valid_numeric(self):
        """Test validation passes with all numeric values."""
        data = {'a': 1, 'b': 2.5, 'c': 0}
        validate_numeric(data)  # Should not raise

    def test_string_value(self):
        """Test error when value is a string."""
        data = {'a': 1, 'b': 'not a number'}
        with self.assertRaises(ValidationError) as ctx:
            validate_numeric(data)
        self.assertIn("Non-numeric", str(ctx.exception))
        self.assertIn("'b'", str(ctx.exception))
        self.assertIn("str", str(ctx.exception))


class TestValidateBounds(unittest.TestCase):
    """Tests for validate_bounds function."""

    def test_within_bounds(self):
        """Test validation passes when all values in bounds."""
        data = {'a': 5, 'b': 10, 'c': 15}
        validate_bounds(data, lower=0, upper=20)  # Should not raise

    def test_below_lower(self):
        """Test error when value below lower bound."""
        data = {'a': 5, 'b': -1}
        with self.assertRaises(ValidationError) as ctx:
            validate_bounds(data, lower=0)
        self.assertIn("below lower bound", str(ctx.exception))
        self.assertIn("'b'", str(ctx.exception))

    def test_above_upper(self):
        """Test error when value above upper bound."""
        data = {'a': 5, 'b': 100}
        with self.assertRaises(ValidationError) as ctx:
            validate_bounds(data, upper=50)
        self.assertIn("above upper bound", str(ctx.exception))
        self.assertIn("'b'", str(ctx.exception))


class TestValidateParameter(unittest.TestCase):
    """Tests for validate_parameter function."""

    def test_valid_parameter(self):
        """Test validation passes for valid parameter."""
        idx = Set(['a', 'b', 'c'], name='letters')
        param = Parameter(idx, data={'a': 1, 'b': 2, 'c': 3})
        validate_parameter(param)  # Should not raise

    def test_parameter_no_data(self):
        """Test error when parameter has no data."""
        idx = Set(['a', 'b', 'c'], name='letters')
        param = Parameter(idx, name='test_param')
        with self.assertRaises(ValidationError) as ctx:
            validate_parameter(param)
        self.assertIn("no data set", str(ctx.exception))

    def test_parameter_bounds(self):
        """Test parameter validation with bounds."""
        idx = Set(['a', 'b', 'c'], name='letters')
        param = Parameter(idx, data={'a': 1, 'b': -5, 'c': 3})
        with self.assertRaises(ValidationError) as ctx:
            validate_parameter(param, lower=0)
        self.assertIn("below lower bound", str(ctx.exception))


class TestSetOperations(unittest.TestCase):
    """Tests for Set operations (union, intersection, etc.)."""

    def test_union(self):
        """Test Set union."""
        A = Set(['a', 'b', 'c'], name='A')
        B = Set(['b', 'c', 'd'], name='B')
        result = A | B
        self.assertEqual(list(result), ['a', 'b', 'c', 'd'])

    def test_intersection(self):
        """Test Set intersection."""
        A = Set(['a', 'b', 'c'], name='A')
        B = Set(['b', 'c', 'd'], name='B')
        result = A & B
        self.assertEqual(list(result), ['b', 'c'])

    def test_difference(self):
        """Test Set difference."""
        A = Set(['a', 'b', 'c'], name='A')
        B = Set(['b', 'c', 'd'], name='B')
        result = A - B
        self.assertEqual(list(result), ['a'])

    def test_symmetric_difference(self):
        """Test Set symmetric difference."""
        A = Set(['a', 'b', 'c'], name='A')
        B = Set(['b', 'c', 'd'], name='B')
        result = A ^ B
        self.assertEqual(list(result), ['a', 'd'])

    def test_subset(self):
        """Test subset checking."""
        A = Set(['a', 'b'], name='A')
        B = Set(['a', 'b', 'c'], name='B')
        self.assertTrue(A <= B)
        self.assertTrue(A < B)
        self.assertFalse(B <= A)

    def test_superset(self):
        """Test superset checking."""
        A = Set(['a', 'b', 'c'], name='A')
        B = Set(['a', 'b'], name='B')
        self.assertTrue(A >= B)
        self.assertTrue(A > B)

    def test_equality(self):
        """Test Set equality."""
        A = Set(['a', 'b', 'c'])
        B = Set(['a', 'b', 'c'])
        C = Set(['a', 'b'])
        self.assertEqual(A, B)
        self.assertNotEqual(A, C)

    def test_filter(self):
        """Test Set filter."""
        nums = Set([1, 2, 3, 4, 5], name='nums')
        evens = nums.filter(lambda x: x % 2 == 0)
        self.assertEqual(list(evens), [2, 4])

    def test_filter_compound(self):
        """Test filter on compound index."""
        routes = Set([('W1', 'C1'), ('W1', 'C2'), ('W2', 'C1')], name='routes')
        w1_routes = routes.filter(lambda r: r[0] == 'W1')
        self.assertEqual(list(w1_routes), [('W1', 'C1'), ('W1', 'C2')])

    def test_map(self):
        """Test Set map."""
        nums = Set([1, 2, 3], name='nums')
        doubled = nums.map(lambda x: x * 2)
        self.assertEqual(list(doubled), [2, 4, 6])

    def test_first_last(self):
        """Test first and last element access."""
        s = Set(['a', 'b', 'c'], name='letters')
        self.assertEqual(s.first(), 'a')
        self.assertEqual(s.last(), 'c')

    def test_first_empty(self):
        """Test first on empty Set raises error."""
        s = Set([], name='empty')
        with self.assertRaises(IndexError):
            s.first()

    def test_sorted(self):
        """Test Set sorting."""
        s = Set([3, 1, 4, 1, 5], name='nums')
        sorted_s = s.sorted()
        self.assertEqual(list(sorted_s), [1, 1, 3, 4, 5])

    def test_sorted_reverse(self):
        """Test Set sorting in reverse."""
        s = Set([3, 1, 4], name='nums')
        sorted_s = s.sorted(reverse=True)
        self.assertEqual(list(sorted_s), [4, 3, 1])

    def test_to_list(self):
        """Test to_list conversion."""
        s = Set(['a', 'b', 'c'])
        self.assertEqual(s.to_list(), ['a', 'b', 'c'])


if __name__ == '__main__':
    unittest.main()
