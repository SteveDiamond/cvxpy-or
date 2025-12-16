"""Tests for cvxpy_or.sets module."""

import unittest

import numpy as np

import cvxpy as cp
from cvxpy_or import (
    Parameter,
    Set,
    Variable,
    sum_by,
    where,
)


class TestSet(unittest.TestCase):
    """Tests for the Set class."""

    def test_simple_index(self):
        """Test basic index with simple elements."""
        idx = Set(['A', 'B', 'C'], name='letters')
        self.assertEqual(len(idx), 3)
        self.assertEqual(list(idx), ['A', 'B', 'C'])
        self.assertIn('A', idx)
        self.assertNotIn('D', idx)
        self.assertEqual(idx.name, 'letters')

    def test_position_lookup(self):
        """Test element-to-position lookup."""
        idx = Set(['X', 'Y', 'Z'])
        self.assertEqual(idx.position('X'), 0)
        self.assertEqual(idx.position('Y'), 1)
        self.assertEqual(idx.position('Z'), 2)

    def test_position_not_found(self):
        """Test error when element not in index."""
        idx = Set(['A', 'B'], name='test')
        with self.assertRaises(KeyError) as ctx:
            idx.position('C')
        self.assertIn('C', str(ctx.exception))
        self.assertIn('test', str(ctx.exception))

    def test_compound_index(self):
        """Test index with tuple elements."""
        idx = Set([('W1', 'C1'), ('W1', 'C2'), ('W2', 'C1')], name='routes')
        self.assertEqual(len(idx), 3)
        self.assertTrue(idx._is_compound)
        self.assertEqual(idx.position(('W1', 'C2')), 1)

    def test_named_positions(self):
        """Test named positions for compound indices."""
        idx = Set(
            [('W1', 'C1'), ('W1', 'C2')],
            name='routes',
            names=('origin', 'destination')
        )
        self.assertEqual(idx.names, ('origin', 'destination'))
        self.assertEqual(idx._resolve_position('origin'), 0)
        self.assertEqual(idx._resolve_position('destination'), 1)
        self.assertEqual(idx._resolve_position(0), 0)

    def test_invalid_position_name(self):
        """Test error for invalid position name."""
        idx = Set([('A', 'B')], names=('first', 'second'))
        with self.assertRaises(KeyError) as ctx:
            idx._resolve_position('invalid')
        self.assertIn('invalid', str(ctx.exception))

    def test_names_arity_mismatch(self):
        """Test error when names don't match tuple arity."""
        with self.assertRaises(ValueError) as ctx:
            Set([('A', 'B', 'C')], names=('first', 'second'))
        self.assertIn('3', str(ctx.exception))
        self.assertIn('2', str(ctx.exception))

    def test_repr(self):
        """Test string representation."""
        idx = Set(['A', 'B', 'C'], name='test')
        r = repr(idx)
        self.assertIn('Set', r)
        self.assertIn('test', r)


class TestParameter(unittest.TestCase):
    """Tests for the Parameter class."""

    def test_creation_with_data(self):
        """Test creating parameter with initial data."""
        idx = Set(['A', 'B', 'C'])
        param = Parameter(idx, data={'A': 1.0, 'B': 2.0, 'C': 3.0})
        np.testing.assert_array_equal(param.value, [1.0, 2.0, 3.0])

    def test_creation_without_data(self):
        """Test creating parameter without data."""
        idx = Set(['A', 'B'])
        param = Parameter(idx, name='p')
        self.assertIsNone(param.value)

    def test_set_data(self):
        """Test setting data after creation."""
        idx = Set(['X', 'Y'])
        param = Parameter(idx)
        param.set_data({'X': 10.0, 'Y': 20.0})
        np.testing.assert_array_equal(param.value, [10.0, 20.0])

    def test_is_cvxpy_parameter(self):
        """Test that Parameter IS a cp.Parameter."""
        idx = Set(['A'])
        param = Parameter(idx, data={'A': 5.0})
        self.assertIsInstance(param, cp.Parameter)

    def test_inner_product_with_matmul(self):
        """Test inner product using @ operator."""
        idx = Set(['A', 'B', 'C'])
        param = Parameter(idx, data={'A': 1.0, 'B': 2.0, 'C': 3.0})
        var = Variable(idx)

        expr = param @ var

        # Should be a MulExpression (matmul), not a sum of multiplies
        self.assertEqual(type(expr).__name__, 'MulExpression')
        # Should have exactly one variable
        self.assertEqual(len(expr.variables()), 1)

    def test_get_value(self):
        """Test get_value for named lookup."""
        idx = Set(['A', 'B', 'C'])
        param = Parameter(idx, data={'A': 1.0, 'B': 2.0, 'C': 3.0})
        self.assertEqual(param.get_value('B'), 2.0)

    def test_get_value_not_set(self):
        """Test get_value returns None when not set."""
        idx = Set(['A', 'B'])
        param = Parameter(idx)
        self.assertIsNone(param.get_value('A'))


class TestVariable(unittest.TestCase):
    """Tests for the Variable class."""

    def test_creation(self):
        """Test basic variable creation."""
        idx = Set(['A', 'B', 'C'])
        var = Variable(idx, name='x')
        self.assertEqual(var.shape, (3,))

    def test_is_cvxpy_variable(self):
        """Test that Variable IS a cp.Variable."""
        idx = Set(['A', 'B'])
        var = Variable(idx)
        self.assertIsInstance(var, cp.Variable)

    def test_nonneg(self):
        """Test non-negative constraint."""
        idx = Set(['A', 'B'])
        var = Variable(idx, nonneg=True)
        self.assertTrue(var.is_nonneg())

    def test_getitem(self):
        """Test element access by key."""
        idx = Set([('W1', 'C1'), ('W1', 'C2')])
        var = Variable(idx)
        elem = var[('W1', 'C2')]
        # Should return an indexed expression
        self.assertEqual(elem.shape, ())

    def test_cvxpy_sum(self):
        """Test cp.sum() works on Variable."""
        idx = Set(['A', 'B', 'C'])
        var = Variable(idx)
        s = cp.sum(var)
        self.assertIsInstance(s, cp.Expression)

    def test_sum_by_integer_position(self):
        """Test sum_by with integer position."""
        idx = Set([
            ('W1', 'C1'), ('W1', 'C2'),
            ('W2', 'C1'), ('W2', 'C2')
        ])
        var = Variable(idx)

        result = sum_by(var, 0, index=idx)

        self.assertIsInstance(result, cp.Expression)
        self.assertEqual(result.shape, (2,))

    def test_sum_by_named_position(self):
        """Test sum_by with named position."""
        idx = Set(
            [('W1', 'C1'), ('W1', 'C2'), ('W2', 'C1')],
            names=('origin', 'destination')
        )
        var = Variable(idx)

        result = sum_by(var, 'origin', index=idx)

        self.assertEqual(result.shape, (2,))

    def test_sum_by_aggregation_matrix(self):
        """Test that sum_by builds correct aggregation matrix."""
        idx = Set([
            ('W1', 'C1'), ('W1', 'C2'),
            ('W2', 'C1'), ('W2', 'C2')
        ])
        var = Variable(idx)

        result = sum_by(var, 0, index=idx)

        # The expression should be a matmul (sparse @ var)
        self.assertEqual(type(result).__name__, 'MulExpression')

    def test_sum_by_on_simple_index_error(self):
        """Test error when sum_by called on non-compound index."""
        idx = Set(['A', 'B', 'C'], name='simple')
        var = Variable(idx)

        with self.assertRaises(ValueError) as ctx:
            sum_by(var, 0, index=idx)
        self.assertIn('compound', str(ctx.exception))
        self.assertIn('simple', str(ctx.exception))

    def test_get_value(self):
        """Test get_value after solving."""
        idx = Set(['A', 'B'])
        var = Variable(idx, nonneg=True)

        prob = cp.Problem(cp.Minimize(cp.sum(var)), [var >= 1])
        prob.solve()

        self.assertAlmostEqual(var.get_value('A'), 1.0, places=4)
        self.assertAlmostEqual(var.get_value('B'), 1.0, places=4)

    def test_get_value_before_solve(self):
        """Test get_value returns None before solving."""
        idx = Set(['A', 'B'])
        var = Variable(idx)
        self.assertIsNone(var.get_value('A'))


class TestCVXPYOperations(unittest.TestCase):
    """Tests for native CVXPY operations on Variable and Parameter."""

    def test_constraint_operators(self):
        """Test constraint operators on Variable."""
        idx = Set(['A', 'B'])
        var = Variable(idx)
        param = Parameter(idx, data={'A': 1.0, 'B': 2.0})

        self.assertIsInstance(var >= 0, cp.Constraint)
        self.assertIsInstance(var <= param, cp.Constraint)
        self.assertIsInstance(var == 0, cp.Constraint)

    def test_cvxpy_atoms(self):
        """Test CVXPY atoms work on Variable."""
        idx = Set(['A', 'B', 'C'])
        var = Variable(idx)

        # All these should work natively
        self.assertIsInstance(cp.abs(var), cp.Expression)
        self.assertIsInstance(cp.sum(var), cp.Expression)
        self.assertIsInstance(cp.sum_squares(var), cp.Expression)
        self.assertIsInstance(cp.norm(var), cp.Expression)

    def test_element_wise_multiply(self):
        """Test cp.multiply() works."""
        idx = Set(['A', 'B', 'C'])
        var = Variable(idx)
        param = Parameter(idx, data={'A': 1.0, 'B': 2.0, 'C': 3.0})

        result = cp.multiply(param, var)
        self.assertIsInstance(result, cp.Expression)
        self.assertEqual(result.shape, (3,))

    def test_arithmetic_operations(self):
        """Test arithmetic operations return CVXPY expressions."""
        idx = Set(['A', 'B'])
        v1 = Variable(idx)
        v2 = Variable(idx)
        p = Parameter(idx, data={'A': 1.0, 'B': 2.0})

        # Variable arithmetic
        self.assertIsInstance(v1 + v2, cp.Expression)
        self.assertIsInstance(v1 - v2, cp.Expression)
        self.assertIsInstance(2 * v1, cp.Expression)

        # Parameter @ Variable
        self.assertIsInstance(p @ v1, cp.Expression)


class TestTransportationProblem(unittest.TestCase):
    """End-to-end test with a transportation problem."""

    def test_transportation_problem_solves(self):
        """Test that a full transportation problem solves correctly."""
        # Define index sets
        warehouses = Set(['W1', 'W2', 'W3'], name='warehouses')
        customers = Set(['C1', 'C2', 'C3'], name='customers')
        routes = Set(
            [('W1', 'C1'), ('W1', 'C2'), ('W1', 'C3'),
             ('W2', 'C1'), ('W2', 'C2'), ('W2', 'C3'),
             ('W3', 'C1'), ('W3', 'C2'), ('W3', 'C3')],
            name='routes',
            names=('origin', 'destination')
        )

        # Parameters
        cost = Parameter(routes, data={
            ('W1', 'C1'): 8, ('W1', 'C2'): 6, ('W1', 'C3'): 10,
            ('W2', 'C1'): 9, ('W2', 'C2'): 4, ('W2', 'C3'): 5,
            ('W3', 'C1'): 14, ('W3', 'C2'): 8, ('W3', 'C3'): 6,
        })
        supply = Parameter(warehouses, data={'W1': 100, 'W2': 150, 'W3': 80})
        demand = Parameter(customers, data={'C1': 80, 'C2': 120, 'C3': 100})

        # Variable
        ship = Variable(routes, nonneg=True, name='ship')

        # Build problem - using native @ operator
        objective = cp.Minimize(cost @ ship)
        constraints = [
            sum_by(ship, 'origin', index=routes) <= supply,
            sum_by(ship, 'destination', index=routes) >= demand,
        ]

        prob = cp.Problem(objective, constraints)
        result = prob.solve()

        # Problem should be feasible and optimal
        self.assertEqual(prob.status, cp.OPTIMAL)

        # Check optimal value (known from solving)
        self.assertAlmostEqual(result, 1690.0, places=2)

        # Check that demand is satisfied
        ship_values = ship.value
        self.assertIsNotNone(ship_values)
        self.assertTrue(np.all(ship_values >= -1e-6))

    def test_sparse_routes(self):
        """Test with sparse route set (not all warehouse-customer pairs)."""
        warehouses = Set(['W1', 'W2'], name='warehouses')
        customers = Set(['C1', 'C2', 'C3'], name='customers')

        # Only some routes exist
        routes = Set(
            [('W1', 'C1'), ('W1', 'C2'),  # W1 can't reach C3
             ('W2', 'C2'), ('W2', 'C3')],  # W2 can't reach C1
            name='routes',
            names=('origin', 'destination')
        )

        cost = Parameter(routes, data={
            ('W1', 'C1'): 5, ('W1', 'C2'): 8,
            ('W2', 'C2'): 6, ('W2', 'C3'): 4,
        })
        supply = Parameter(warehouses, data={'W1': 100, 'W2': 100})
        demand = Parameter(customers, data={'C1': 50, 'C2': 60, 'C3': 40})

        ship = Variable(routes, nonneg=True)

        prob = cp.Problem(
            cp.Minimize(cost @ ship),
            [
                sum_by(ship, 'origin', index=routes) <= supply,
                sum_by(ship, 'destination', index=routes) >= demand,
            ]
        )
        prob.solve()

        self.assertEqual(prob.status, cp.OPTIMAL)


class TestEfficiency(unittest.TestCase):
    """Tests to verify efficient expression emission."""

    def test_matmul_is_single_expression(self):
        """Verify @ creates single matmul, not sum of multiplies."""
        idx = Set(['A', 'B', 'C', 'D', 'E'])
        param = Parameter(idx, data={k: i for i, k in enumerate(idx)})
        var = Variable(idx)

        expr = param @ var

        # Should be a MulExpression
        self.assertEqual(type(expr).__name__, 'MulExpression')

    def test_sum_by_is_sparse_matmul(self):
        """Verify sum_by uses sparse matrix multiplication."""
        idx = Set([
            ('A', 1), ('A', 2), ('A', 3),
            ('B', 1), ('B', 2), ('B', 3),
        ])
        var = Variable(idx)

        result = sum_by(var, 0, index=idx)

        # Should be sparse @ var, which is a MulExpression
        self.assertEqual(type(result).__name__, 'MulExpression')


class TestSetCross(unittest.TestCase):
    """Tests for Set.cross() cross-product functionality."""

    def test_cross_two_indices(self):
        """Test cross product of two indices."""
        warehouses = Set(['W1', 'W2'], name='warehouses')
        customers = Set(['C1', 'C2', 'C3'], name='customers')

        routes = Set.cross(warehouses, customers)

        self.assertEqual(len(routes), 6)
        self.assertEqual(
            list(routes),
            [('W1', 'C1'), ('W1', 'C2'), ('W1', 'C3'),
             ('W2', 'C1'), ('W2', 'C2'), ('W2', 'C3')]
        )

    def test_cross_auto_names(self):
        """Test that cross auto-generates position names."""
        warehouses = Set(['W1', 'W2'], name='warehouses')
        customers = Set(['C1', 'C2'], name='customers')

        routes = Set.cross(warehouses, customers)

        self.assertEqual(routes.names, ('warehouses', 'customers'))

    def test_cross_explicit_names(self):
        """Test explicit names override auto-generation."""
        a = Set([1, 2], name='a')
        b = Set([3, 4], name='b')

        idx = Set.cross(a, b, names=('first', 'second'))

        self.assertEqual(idx.names, ('first', 'second'))

    def test_cross_three_indices(self):
        """Test cross product of three indices."""
        a = Set(['A', 'B'], name='a')
        b = Set([1, 2], name='b')
        c = Set(['x', 'y'], name='c')

        idx = Set.cross(a, b, c)

        self.assertEqual(len(idx), 8)  # 2 * 2 * 2
        self.assertIn(('A', 1, 'x'), idx)
        self.assertIn(('B', 2, 'y'), idx)

    def test_cross_requires_two_indices(self):
        """Test that cross requires at least 2 indices."""
        a = Set(['A', 'B'])

        with self.assertRaises(ValueError):
            Set.cross(a)


class TestParameterExpand(unittest.TestCase):
    """Tests for Parameter.expand() broadcasting."""

    def test_expand_1d_to_2d(self):
        """Test expanding 1D parameter to 2D cross-product index."""
        warehouses = Set(['W1', 'W2'], name='warehouses')
        periods = Set(['T1', 'T2'], name='periods')

        holding_cost = Parameter(warehouses, data={'W1': 0.1, 'W2': 0.2})
        inv_idx = Set.cross(warehouses, periods)

        expanded = holding_cost.expand(inv_idx, ['warehouses'])

        # Check values are correctly broadcast
        np.testing.assert_array_almost_equal(
            expanded.value,
            [0.1, 0.1, 0.2, 0.2]  # (W1,T1), (W1,T2), (W2,T1), (W2,T2)
        )

    def test_expand_2d_to_3d(self):
        """Test expanding 2D parameter to 3D cross-product index."""
        warehouses = Set(['W1', 'W2'], name='warehouses')
        customers = Set(['C1', 'C2'], name='customers')
        periods = Set(['T1', 'T2'], name='periods')

        routes = Set.cross(warehouses, customers)
        cost = Parameter(routes, data={
            ('W1', 'C1'): 10, ('W1', 'C2'): 20,
            ('W2', 'C1'): 30, ('W2', 'C2'): 40,
        })

        shipments = Set.cross(warehouses, customers, periods)
        expanded = cost.expand(shipments, ['warehouses', 'customers'])

        # Check values: each (w,c) cost is repeated for each period
        expected = [
            10, 10,  # (W1,C1,T1), (W1,C1,T2)
            20, 20,  # (W1,C2,T1), (W1,C2,T2)
            30, 30,  # (W2,C1,T1), (W2,C1,T2)
            40, 40,  # (W2,C2,T1), (W2,C2,T2)
        ]
        np.testing.assert_array_almost_equal(expanded.value, expected)

    def test_expand_with_integer_positions(self):
        """Test expand with integer position indices."""
        warehouses = Set(['W1', 'W2'], name='warehouses')
        periods = Set(['T1', 'T2'], name='periods')

        holding_cost = Parameter(warehouses, data={'W1': 0.1, 'W2': 0.2})
        inv_idx = Set.cross(warehouses, periods)

        expanded = holding_cost.expand(inv_idx, [0])

        np.testing.assert_array_almost_equal(
            expanded.value,
            [0.1, 0.1, 0.2, 0.2]
        )

    def test_expand_inner_product(self):
        """Test that expanded parameter can be used in @ operator."""
        warehouses = Set(['W1', 'W2'], name='warehouses')
        periods = Set(['T1', 'T2'], name='periods')

        holding_cost = Parameter(warehouses, data={'W1': 1.0, 'W2': 2.0})
        inv_idx = Set.cross(warehouses, periods)
        inv = Variable(inv_idx, nonneg=True)

        expanded = holding_cost.expand(inv_idx, ['warehouses'])
        expr = expanded @ inv

        # Should be a valid CVXPY expression
        self.assertEqual(type(expr).__name__, 'MulExpression')


class TestMultiPositionSumBy(unittest.TestCase):
    """Tests for sum_by with multiple positions."""

    def test_sum_by_two_positions(self):
        """Test sum_by with two positions (3D -> 2D)."""
        warehouses = Set(['W1', 'W2'], name='warehouses')
        customers = Set(['C1', 'C2'], name='customers')
        periods = Set(['T1', 'T2'], name='periods')

        idx = Set.cross(warehouses, customers, periods)
        var = Variable(idx)

        # Sum over customers, keeping (warehouse, period)
        result = sum_by(var, ['warehouses', 'periods'], index=idx)

        self.assertIsInstance(result, cp.Expression)
        self.assertEqual(result.shape, (4,))  # 2 warehouses x 2 periods

    def test_sum_by_two_positions_integers(self):
        """Test sum_by with integer positions."""
        warehouses = Set(['W1', 'W2'], name='warehouses')
        customers = Set(['C1', 'C2'], name='customers')
        periods = Set(['T1', 'T2'], name='periods')

        idx = Set.cross(warehouses, customers, periods)
        var = Variable(idx)

        # Sum over customers (position 1), keeping positions 0 and 2
        result = sum_by(var, [0, 2], index=idx)

        self.assertEqual(result.shape, (4,))

    def test_sum_by_non_adjacent_positions(self):
        """Test sum_by with non-adjacent positions."""
        a = Set(['A1', 'A2'], name='a')
        b = Set(['B1', 'B2'], name='b')
        c = Set(['C1', 'C2'], name='c')

        idx = Set.cross(a, b, c)
        var = Variable(idx)

        # Keep (a, c), sum over b
        result = sum_by(var, ['a', 'c'], index=idx)

        self.assertEqual(result.shape, (4,))

    def test_multi_period_transportation_no_loops(self):
        """Full test of multi-period transportation without for loops."""
        warehouses = Set(['W1', 'W2'], name='warehouses')
        customers = Set(['C1', 'C2'], name='customers')
        periods = Set(['T1', 'T2'], name='periods')

        shipments = Set.cross(warehouses, customers, periods)
        ship = Variable(shipments, nonneg=True)

        # Supply indexed by (warehouse, period)
        supply_idx = Set.cross(warehouses, periods)
        supply = Parameter(supply_idx, data={
            ('W1', 'T1'): 100, ('W1', 'T2'): 100,
            ('W2', 'T1'): 100, ('W2', 'T2'): 100,
        })

        # Demand indexed by (customer, period)
        demand_idx = Set.cross(customers, periods)
        demand = Parameter(demand_idx, data={
            ('C1', 'T1'): 60, ('C1', 'T2'): 60,
            ('C2', 'T1'): 70, ('C2', 'T2'): 70,
        })

        # NO FOR LOOPS! Constraints using multi-position sum_by
        constraints = [
            # Sum over customers for each (warehouse, period)
            sum_by(ship, ['warehouses', 'periods'], index=shipments) <= supply,
            # Sum over warehouses for each (customer, period)
            sum_by(ship, ['customers', 'periods'], index=shipments) >= demand,
        ]

        # Simple objective
        objective = cp.Minimize(cp.sum(ship))

        prob = cp.Problem(objective, constraints)
        result = prob.solve()

        self.assertEqual(prob.status, cp.OPTIMAL)
        self.assertAlmostEqual(result, 260.0, places=1)


class TestCrossProductProblem(unittest.TestCase):
    """End-to-end test using Set.cross()."""

    def test_transportation_with_cross(self):
        """Test transportation problem using Set.cross()."""
        warehouses = Set(['W1', 'W2'], name='warehouses')
        customers = Set(['C1', 'C2'], name='customers')
        routes = Set.cross(warehouses, customers, name='routes')

        cost = Parameter(routes, data={
            ('W1', 'C1'): 8, ('W1', 'C2'): 6,
            ('W2', 'C1'): 9, ('W2', 'C2'): 4,
        })
        supply = Parameter(warehouses, data={'W1': 100, 'W2': 100})
        demand = Parameter(customers, data={'C1': 80, 'C2': 80})

        ship = Variable(routes, nonneg=True)

        prob = cp.Problem(
            cp.Minimize(cost @ ship),
            [
                sum_by(ship, 'warehouses', index=routes) <= supply,
                sum_by(ship, 'customers', index=routes) >= demand,
            ]
        )
        prob.solve()

        self.assertEqual(prob.status, cp.OPTIMAL)

    def test_combined_variables(self):
        """Test arithmetic on Variables."""
        warehouses = Set(['W1', 'W2'], name='warehouses')
        customers = Set(['C1', 'C2'], name='customers')
        routes = Set.cross(warehouses, customers)

        ship = Variable(routes, nonneg=True)
        express = Variable(routes, nonneg=True)

        # Combine variables - returns cp.Expression
        total = ship + express
        self.assertIsInstance(total, cp.Expression)


class TestWhere(unittest.TestCase):
    """Tests for where() function."""

    def test_where_boolean_mask(self):
        """Test where() with boolean array."""
        idx = Set(['A', 'B', 'C'], name='items')
        var = Variable(idx, nonneg=True)
        mask = np.array([True, False, True])
        expr = where(var, mask)
        self.assertIsInstance(expr, cp.Expression)

    def test_where_on_expression(self):
        """Test where() works on arbitrary expressions."""
        idx = Set(['A', 'B', 'C'], name='items')
        var = Variable(idx, nonneg=True)
        expr = 2 * var + 1
        mask = np.array([True, False, True])
        filtered = where(expr, mask)
        self.assertIsInstance(filtered, cp.Expression)

    def test_where_callable(self):
        """Test where() with callable and index."""
        routes = Set(
            [('W1', 'C1'), ('W1', 'C2'), ('W2', 'C1')],
            names=('origin', 'dest')
        )
        var = Variable(routes, nonneg=True)
        expr = where(var, lambda r: r[0] == 'W1', index=routes)
        self.assertIsInstance(expr, cp.Expression)

    def test_where_kwargs(self):
        """Test where() with keyword filtering."""
        routes = Set(
            [('W1', 'C1'), ('W1', 'C2'), ('W2', 'C1')],
            names=('origin', 'dest')
        )
        var = Variable(routes, nonneg=True)
        expr = where(var, index=routes, origin='W1')
        self.assertIsInstance(expr, cp.Expression)

    def test_where_kwargs_list(self):
        """Test where() with list of allowed values."""
        routes = Set(
            [('W1', 'C1'), ('W2', 'C2'), ('W3', 'C1')],
            names=('origin', 'dest')
        )
        var = Variable(routes, nonneg=True)
        expr = where(var, index=routes, origin=['W1', 'W2'])
        self.assertIsInstance(expr, cp.Expression)

    def test_where_callable_requires_index(self):
        """Test error when callable used without index."""
        idx = Set(['A', 'B', 'C'], name='items')
        var = Variable(idx, nonneg=True)
        with self.assertRaises(ValueError) as ctx:
            where(var, lambda x: True)
        self.assertIn('index', str(ctx.exception).lower())

    def test_where_kwargs_requires_index(self):
        """Test error when kwargs used without index."""
        routes = Set([('W1', 'C1')], names=('origin', 'dest'))
        var = Variable(routes, nonneg=True)
        with self.assertRaises(ValueError) as ctx:
            where(var, origin='W1')
        self.assertIn('index', str(ctx.exception).lower())

    def test_where_simple_index_rejects_kwargs(self):
        """Test error when using kwargs on simple index."""
        idx = Set(['A', 'B', 'C'], name='simple')
        var = Variable(idx)
        with self.assertRaises(ValueError) as ctx:
            where(var, index=idx, foo='bar')
        self.assertIn('compound', str(ctx.exception))

    def test_where_solves_correctly(self):
        """Test that where() produces correct optimization results.

        Use case: minimize cost over valid routes only.
        """
        idx = Set(['A', 'B', 'C'], name='items')
        var = Variable(idx, nonneg=True)
        mask = np.array([1.0, 0.0, 1.0])

        # Minimize masked sum: only A and C contribute to objective
        # Constraint: all vars >= 1
        prob = cp.Problem(
            cp.Minimize(cp.sum(where(var, mask))),  # Only count A and C
            [var >= 1]
        )
        prob.solve()

        self.assertEqual(prob.status, cp.OPTIMAL)
        # A and C should be exactly 1 (minimized in objective)
        self.assertAlmostEqual(var.value[0], 1.0, places=4)
        self.assertAlmostEqual(var.value[2], 1.0, places=4)
        # B is unconstrained in objective, just needs >= 1
        self.assertGreaterEqual(var.value[1], 1.0 - 1e-4)
        # Objective should be 2 (A + C only)
        self.assertAlmostEqual(prob.value, 2.0, places=4)

    def test_where_expression_solves_correctly(self):
        """Test where() on expression produces correct results.

        Use case: minimize expression over valid entries only.
        """
        idx = Set(['A', 'B', 'C'], name='items')
        var = Variable(idx, nonneg=True)
        mask = np.array([1.0, 0.0, 1.0])

        # Minimize masked expression: only A and C contribute
        prob = cp.Problem(
            cp.Minimize(cp.sum(where(2 * var + 1, mask))),  # 2*A + 1 + 2*C + 1
            [var >= 1]
        )
        prob.solve()

        self.assertEqual(prob.status, cp.OPTIMAL)
        # A and C minimized in objective
        self.assertAlmostEqual(var.value[0], 1.0, places=4)
        self.assertAlmostEqual(var.value[2], 1.0, places=4)
        # B unconstrained in objective
        self.assertGreaterEqual(var.value[1], 1.0 - 1e-4)
        # Objective: 2*1 + 1 + 0 + 2*1 + 1 = 6
        self.assertAlmostEqual(prob.value, 6.0, places=4)


class TestSumByFunction(unittest.TestCase):
    """Tests for standalone sum_by() function."""

    def test_sum_by_on_expression(self):
        """Test sum_by() works on arbitrary expressions."""
        routes = Set(
            [('W1', 'C1'), ('W1', 'C2'), ('W2', 'C1'), ('W2', 'C2')],
            names=('origin', 'dest')
        )
        var = Variable(routes, nonneg=True)
        expr = 2 * var + 1
        result = sum_by(expr, 'origin', index=routes)
        self.assertIsInstance(result, cp.Expression)
        self.assertEqual(result.shape, (2,))


if __name__ == '__main__':
    unittest.main()
