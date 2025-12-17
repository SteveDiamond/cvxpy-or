"""Tests for cvxpy_or.aggregations module."""

import unittest

import cvxpy as cp
import numpy as np

from cvxpy_or import (
    Parameter,
    Set,
    Variable,
    count_by,
    group_keys,
    max_by,
    mean_by,
    min_by,
    sum_by,
)


class TestMeanBy(unittest.TestCase):
    """Tests for mean_by function."""

    def test_mean_by_single_position(self):
        """Test mean_by with a single position."""
        warehouses = Set(["W1", "W2"], name="warehouses")
        customers = Set(["C1", "C2", "C3"], name="customers")
        routes = Set.cross(warehouses, customers, name="routes")

        # Create parameter with known values
        param = Parameter(
            routes,
            data={
                ("W1", "C1"): 10,
                ("W1", "C2"): 20,
                ("W1", "C3"): 30,
                ("W2", "C1"): 40,
                ("W2", "C2"): 50,
                ("W2", "C3"): 60,
            },
        )

        # Mean by warehouse
        result = mean_by(param, "warehouses")
        self.assertEqual(result.shape, (2,))

        # Evaluate (parameter, so it's a constant)
        expected = np.array([20.0, 50.0])  # (10+20+30)/3, (40+50+60)/3
        np.testing.assert_array_almost_equal(result.value, expected)

    def test_mean_by_in_optimization(self):
        """Test mean_by works in optimization problem."""
        warehouses = Set(["W1", "W2"], name="warehouses")
        customers = Set(["C1", "C2"], name="customers")
        routes = Set.cross(warehouses, customers, name="routes")

        ship = Variable(routes, nonneg=True, name="ship")
        demand = Parameter(
            routes,
            data={
                ("W1", "C1"): 10,
                ("W1", "C2"): 20,
                ("W2", "C1"): 30,
                ("W2", "C2"): 40,
            },
        )

        # Problem: maximize mean shipment per warehouse, subject to demand
        prob = cp.Problem(cp.Maximize(cp.sum(mean_by(ship, "warehouses"))), [ship <= demand])
        prob.solve()

        self.assertEqual(prob.status, cp.OPTIMAL)
        # Should ship full demand
        np.testing.assert_array_almost_equal(ship.value, demand.value)


class TestCountBy(unittest.TestCase):
    """Tests for count_by function."""

    def test_count_by(self):
        """Test counting elements per group."""
        warehouses = Set(["W1", "W2", "W3"], name="warehouses")
        customers = Set(["C1", "C2"], name="customers")
        routes = Set.cross(warehouses, customers, name="routes")

        # Each warehouse serves 2 customers
        counts = count_by(routes, "warehouses")
        np.testing.assert_array_equal(counts, [2, 2, 2])

        # Each customer is served by 3 warehouses
        counts = count_by(routes, "customers")
        np.testing.assert_array_equal(counts, [3, 3])


class TestGroupKeys(unittest.TestCase):
    """Tests for group_keys function."""

    def test_group_keys_single_position(self):
        """Test getting unique keys for single position."""
        warehouses = Set(["W1", "W2"], name="warehouses")
        customers = Set(["C1", "C2", "C3"], name="customers")
        routes = Set.cross(warehouses, customers, name="routes")

        keys = group_keys(routes, "warehouses")
        self.assertEqual(keys, ["W1", "W2"])

        keys = group_keys(routes, "customers")
        self.assertEqual(keys, ["C1", "C2", "C3"])

    def test_group_keys_multiple_positions(self):
        """Test getting unique keys for multiple positions."""
        w = Set(["W1", "W2"], name="w")
        c = Set(["C1", "C2"], name="c")
        t = Set(["T1", "T2"], name="t")
        routes = Set.cross(w, c, t, name="routes")

        keys = group_keys(routes, ["w", "c"])
        self.assertEqual(keys, [("W1", "C1"), ("W1", "C2"), ("W2", "C1"), ("W2", "C2")])


class TestMaxBy(unittest.TestCase):
    """Tests for max_by function."""

    def test_max_by_returns_var_and_constraints(self):
        """Test max_by returns variable and constraints."""
        warehouses = Set(["W1", "W2"], name="warehouses")
        customers = Set(["C1", "C2"], name="customers")
        routes = Set.cross(warehouses, customers, name="routes")

        ship = Variable(routes, nonneg=True)
        max_ship, constraints = max_by(ship, "warehouses")

        self.assertIsInstance(max_ship, cp.Variable)
        self.assertEqual(max_ship.shape, (2,))  # 2 warehouses
        self.assertEqual(len(constraints), 4)  # 4 routes

    def test_max_by_in_optimization(self):
        """Test max_by works in optimization."""
        warehouses = Set(["W1", "W2"], name="warehouses")
        customers = Set(["C1", "C2"], name="customers")
        routes = Set.cross(warehouses, customers, name="routes")

        ship = Variable(routes, nonneg=True)
        _cost = Parameter(
            routes,
            data={
                ("W1", "C1"): 1,
                ("W1", "C2"): 2,
                ("W2", "C1"): 3,
                ("W2", "C2"): 4,
            },
        )  # noqa: F841

        # Minimize max shipment across all routes
        max_ship, constraints = max_by(ship, "warehouses")

        prob = cp.Problem(
            cp.Minimize(cp.sum(max_ship)),
            constraints
            + [
                sum_by(ship, "customers") >= 10,  # Demand
            ],
        )
        prob.solve()

        self.assertEqual(prob.status, cp.OPTIMAL)


class TestMinBy(unittest.TestCase):
    """Tests for min_by function."""

    def test_min_by_returns_var_and_constraints(self):
        """Test min_by returns variable and constraints."""
        warehouses = Set(["W1", "W2"], name="warehouses")
        customers = Set(["C1", "C2"], name="customers")
        routes = Set.cross(warehouses, customers, name="routes")

        ship = Variable(routes, nonneg=True)
        min_ship, constraints = min_by(ship, "warehouses")

        self.assertIsInstance(min_ship, cp.Variable)
        self.assertEqual(min_ship.shape, (2,))
        self.assertEqual(len(constraints), 4)

    def test_min_by_in_optimization(self):
        """Test min_by works to maximize minimum."""
        warehouses = Set(["W1", "W2"], name="warehouses")
        customers = Set(["C1", "C2"], name="customers")
        routes = Set.cross(warehouses, customers, name="routes")

        ship = Variable(routes, nonneg=True)

        # Max-min fairness: maximize the minimum shipment to any customer
        min_ship, constraints = min_by(ship, "customers")

        prob = cp.Problem(
            cp.Maximize(cp.sum(min_ship)),
            constraints
            + [
                sum_by(ship, "warehouses") <= 100,  # Supply limit
            ],
        )
        prob.solve()

        self.assertEqual(prob.status, cp.OPTIMAL)


class TestAggregationErrors(unittest.TestCase):
    """Tests for aggregation error handling."""

    def test_mean_by_simple_index_error(self):
        """Test error when using mean_by on simple index."""
        idx = Set(["a", "b", "c"], name="simple")
        param = Parameter(idx, data={"a": 1, "b": 2, "c": 3})

        with self.assertRaises(ValueError) as ctx:
            mean_by(param, 0)
        self.assertIn("compound index", str(ctx.exception))

    def test_count_by_simple_index_error(self):
        """Test error when using count_by on simple index."""
        idx = Set(["a", "b", "c"], name="simple")

        with self.assertRaises(ValueError) as ctx:
            count_by(idx, 0)
        self.assertIn("compound index", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
