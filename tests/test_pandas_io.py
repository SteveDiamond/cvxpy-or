"""Tests for pandas I/O functionality."""

from __future__ import annotations

import unittest

import cvxpy as cp
import numpy as np
import pandas as pd

from cvxpy_or import (
    Parameter,
    Set,
    Variable,
    sum_by,
)
from cvxpy_or.pandas_io import (
    parameter_from_dataframe,
    parameter_from_series,
    parameter_to_dataframe,
    set_from_dataframe,
    set_from_index,
    set_from_series,
    variable_to_dataframe,
)


class TestSetFromDataFrame(unittest.TestCase):
    """Tests for set_from_dataframe."""

    def test_2d_set(self):
        """Test creating 2D Set from DataFrame."""
        df = pd.DataFrame(
            {
                "warehouse": ["W1", "W1", "W2", "W2"],
                "customer": ["C1", "C2", "C1", "C2"],
                "cost": [10, 15, 20, 25],
            }
        )
        routes = set_from_dataframe(df, ["warehouse", "customer"], name="routes")
        self.assertEqual(len(routes), 4)
        self.assertTrue(routes._is_compound)
        self.assertEqual(routes.names, ("warehouse", "customer"))
        self.assertIn(("W1", "C1"), routes)
        self.assertIn(("W2", "C2"), routes)

    def test_3d_set(self):
        """Test creating 3D Set from DataFrame."""
        df = pd.DataFrame(
            {
                "warehouse": ["W1", "W1", "W1", "W1", "W2", "W2", "W2", "W2"],
                "customer": ["C1", "C1", "C2", "C2", "C1", "C1", "C2", "C2"],
                "period": ["T1", "T2", "T1", "T2", "T1", "T2", "T1", "T2"],
                "cost": [10, 12, 15, 14, 20, 22, 18, 19],
            }
        )
        shipments = set_from_dataframe(df, ["warehouse", "customer", "period"], name="shipments")
        self.assertEqual(len(shipments), 8)
        self.assertTrue(shipments._is_compound)
        self.assertEqual(shipments.names, ("warehouse", "customer", "period"))
        self.assertIn(("W1", "C1", "T1"), shipments)
        self.assertIn(("W2", "C2", "T2"), shipments)

    def test_4d_set(self):
        """Test creating 4D Set from DataFrame."""
        # 2 warehouses x 2 customers x 2 periods x 2 products = 16 elements
        rows = []
        for w in ["W1", "W2"]:
            for c in ["C1", "C2"]:
                for t in ["T1", "T2"]:
                    for p in ["P1", "P2"]:
                        rows.append(
                            {
                                "warehouse": w,
                                "customer": c,
                                "period": t,
                                "product": p,
                                "cost": np.random.rand(),
                            }
                        )
        df = pd.DataFrame(rows)

        idx = set_from_dataframe(
            df, ["warehouse", "customer", "period", "product"], name="allocations"
        )
        self.assertEqual(len(idx), 16)
        self.assertTrue(idx._is_compound)
        self.assertEqual(idx.names, ("warehouse", "customer", "period", "product"))
        self.assertIn(("W1", "C1", "T1", "P1"), idx)
        self.assertIn(("W2", "C2", "T2", "P2"), idx)

    def test_custom_names(self):
        """Test custom position names for Set."""
        df = pd.DataFrame(
            {
                "src": ["A", "A", "B"],
                "dst": ["X", "Y", "X"],
            }
        )
        s = set_from_dataframe(df, ["src", "dst"], names=("origin", "destination"))
        self.assertEqual(s.names, ("origin", "destination"))


class TestParameterFromDataFrame(unittest.TestCase):
    """Tests for parameter_from_dataframe."""

    def test_2d_parameter(self):
        """Test creating 2D Parameter from DataFrame."""
        df = pd.DataFrame(
            {
                "warehouse": ["W1", "W1", "W2", "W2"],
                "customer": ["C1", "C2", "C1", "C2"],
                "cost": [10, 15, 20, 25],
            }
        )
        cost = parameter_from_dataframe(df, ["warehouse", "customer"], "cost")
        self.assertEqual(cost.shape, (4,))
        self.assertEqual(cost.get_value(("W1", "C1")), 10)
        self.assertEqual(cost.get_value(("W2", "C2")), 25)

    def test_3d_parameter(self):
        """Test creating 3D Parameter from DataFrame."""
        df = pd.DataFrame(
            {
                "warehouse": ["W1", "W1", "W1", "W1", "W2", "W2", "W2", "W2"],
                "customer": ["C1", "C1", "C2", "C2", "C1", "C1", "C2", "C2"],
                "period": ["T1", "T2", "T1", "T2", "T1", "T2", "T1", "T2"],
                "cost": [10, 12, 15, 14, 20, 22, 18, 19],
            }
        )
        cost = parameter_from_dataframe(
            df, ["warehouse", "customer", "period"], "cost", name="cost"
        )
        self.assertEqual(cost.shape, (8,))
        self.assertEqual(cost.get_value(("W1", "C1", "T1")), 10)
        self.assertEqual(cost.get_value(("W1", "C1", "T2")), 12)
        self.assertEqual(cost.get_value(("W2", "C2", "T2")), 19)

    def test_4d_parameter(self):
        """Test creating 4D Parameter from DataFrame."""
        rows = []
        expected = {}
        for w in ["W1", "W2"]:
            for c in ["C1", "C2"]:
                for t in ["T1", "T2"]:
                    for p in ["P1", "P2"]:
                        val = hash((w, c, t, p)) % 100
                        rows.append(
                            {
                                "warehouse": w,
                                "customer": c,
                                "period": t,
                                "product": p,
                                "cost": val,
                            }
                        )
                        expected[(w, c, t, p)] = val
        df = pd.DataFrame(rows)

        cost = parameter_from_dataframe(df, ["warehouse", "customer", "period", "product"], "cost")
        self.assertEqual(cost.shape, (16,))
        # Check a few values
        self.assertEqual(
            cost.get_value(("W1", "C1", "T1", "P1")), expected[("W1", "C1", "T1", "P1")]
        )
        self.assertEqual(
            cost.get_value(("W2", "C2", "T2", "P2")), expected[("W2", "C2", "T2", "P2")]
        )

    def test_with_existing_index(self):
        """Test creating Parameter with pre-existing Set index."""
        warehouses = Set(["W1", "W2"], name="warehouses")
        customers = Set(["C1", "C2"], name="customers")
        periods = Set(["T1", "T2"], name="periods")
        shipments = Set.cross(warehouses, customers, periods)

        df = pd.DataFrame(
            {
                "warehouse": ["W1", "W1", "W1", "W1", "W2", "W2", "W2", "W2"],
                "customer": ["C1", "C1", "C2", "C2", "C1", "C1", "C2", "C2"],
                "period": ["T1", "T2", "T1", "T2", "T1", "T2", "T1", "T2"],
                "cost": [10, 12, 15, 14, 20, 22, 18, 19],
            }
        )
        # Note: index_cols must match Set.cross order (warehouses, customers, periods)
        cost = parameter_from_dataframe(
            df, ["warehouse", "customer", "period"], "cost", index=shipments
        )
        self.assertIs(cost.index, shipments)
        self.assertEqual(cost.get_value(("W1", "C1", "T1")), 10)


class TestVariableToDataFrame(unittest.TestCase):
    """Tests for variable_to_dataframe."""

    def test_3d_variable_export(self):
        """Test exporting 3D Variable to DataFrame."""
        warehouses = Set(["W1", "W2"], name="warehouse")
        customers = Set(["C1", "C2"], name="customer")
        periods = Set(["T1", "T2"], name="period")
        shipments = Set.cross(warehouses, customers, periods)

        ship = Variable(shipments, nonneg=True)

        # Solve simple problem
        prob = cp.Problem(cp.Minimize(cp.sum(ship)), [ship >= 1])
        prob.solve()

        df = variable_to_dataframe(ship)
        self.assertEqual(len(df), 8)
        self.assertListEqual(list(df.columns), ["warehouse", "customer", "period", "value"])
        # Check that all tuples are present
        for w in ["W1", "W2"]:
            for c in ["C1", "C2"]:
                for t in ["T1", "T2"]:
                    row = df[(df["warehouse"] == w) & (df["customer"] == c) & (df["period"] == t)]
                    self.assertEqual(len(row), 1)
                    self.assertAlmostEqual(row["value"].iloc[0], 1.0, places=4)


class TestParameterToDataFrame(unittest.TestCase):
    """Tests for parameter_to_dataframe."""

    def test_3d_parameter_export(self):
        """Test exporting 3D Parameter to DataFrame."""
        warehouses = Set(["W1", "W2"], name="warehouse")
        customers = Set(["C1", "C2"], name="customer")
        periods = Set(["T1", "T2"], name="period")
        shipments = Set.cross(warehouses, customers, periods)

        data = {
            ("W1", "C1", "T1"): 10,
            ("W1", "C1", "T2"): 12,
            ("W1", "C2", "T1"): 15,
            ("W1", "C2", "T2"): 14,
            ("W2", "C1", "T1"): 20,
            ("W2", "C1", "T2"): 22,
            ("W2", "C2", "T1"): 18,
            ("W2", "C2", "T2"): 19,
        }
        param = Parameter(shipments, data=data, name="cost")

        df = parameter_to_dataframe(param)
        self.assertEqual(len(df), 8)
        self.assertListEqual(list(df.columns), ["warehouse", "customer", "period", "value"])

        # Check specific values
        row = df[(df["warehouse"] == "W1") & (df["customer"] == "C1") & (df["period"] == "T1")]
        self.assertEqual(row["value"].iloc[0], 10)


class TestRoundTrip(unittest.TestCase):
    """Tests for round-trip DataFrame conversions."""

    def test_3d_parameter_roundtrip(self):
        """Test DataFrame -> Parameter -> DataFrame for 3D data."""
        original_df = pd.DataFrame(
            {
                "warehouse": ["W1", "W1", "W1", "W1", "W2", "W2", "W2", "W2"],
                "customer": ["C1", "C1", "C2", "C2", "C1", "C1", "C2", "C2"],
                "period": ["T1", "T2", "T1", "T2", "T1", "T2", "T1", "T2"],
                "cost": [10.0, 12.0, 15.0, 14.0, 20.0, 22.0, 18.0, 19.0],
            }
        )

        param = parameter_from_dataframe(original_df, ["warehouse", "customer", "period"], "cost")
        result_df = parameter_to_dataframe(param, value_col="cost")

        # Sort both DataFrames for comparison
        original_sorted = original_df.sort_values(["warehouse", "customer", "period"]).reset_index(
            drop=True
        )
        result_sorted = result_df.sort_values(["warehouse", "customer", "period"]).reset_index(
            drop=True
        )

        pd.testing.assert_frame_equal(original_sorted, result_sorted)


class TestMultiDimensionalTransportation(unittest.TestCase):
    """End-to-end tests with multi-dimensional data from DataFrames."""

    def test_3d_multi_period_transportation(self):
        """Test 3D multi-period transportation problem with pandas input."""
        # Cost data (2D: warehouse x customer)
        cost_df = pd.DataFrame(
            {
                "warehouse": ["W1", "W1", "W2", "W2"],
                "customer": ["C1", "C2", "C1", "C2"],
                "cost": [10.0, 15.0, 20.0, 12.0],
            }
        )

        # Supply data (2D: warehouse x period)
        supply_df = pd.DataFrame(
            {
                "warehouse": ["W1", "W1", "W2", "W2"],
                "period": ["T1", "T2", "T1", "T2"],
                "supply": [100.0, 120.0, 80.0, 90.0],
            }
        )

        # Demand data (2D: customer x period)
        demand_df = pd.DataFrame(
            {
                "customer": ["C1", "C1", "C2", "C2"],
                "period": ["T1", "T2", "T1", "T2"],
                "demand": [60.0, 70.0, 50.0, 55.0],
            }
        )

        # Create 1D sets
        warehouses = set_from_series(cost_df["warehouse"], name="warehouse")
        customers = set_from_series(cost_df["customer"], name="customer")
        periods = set_from_series(supply_df["period"], name="period")

        # Create 3D index and 2D indices
        shipments = Set.cross(warehouses, customers, periods)
        supply_idx = Set.cross(warehouses, periods)
        demand_idx = Set.cross(customers, periods)

        # Create parameters
        cost = parameter_from_dataframe(cost_df, ["warehouse", "customer"], "cost")
        supply = parameter_from_dataframe(
            supply_df, ["warehouse", "period"], "supply", index=supply_idx
        )
        demand = parameter_from_dataframe(
            demand_df, ["customer", "period"], "demand", index=demand_idx
        )

        # Expand 2D cost to 3D
        cost_3d = cost.expand(shipments, ["warehouse", "customer"])

        # Create 3D variable
        ship = Variable(shipments, nonneg=True)

        # Build and solve problem
        prob = cp.Problem(
            cp.Minimize(cost_3d @ ship),
            [
                sum_by(ship, ["warehouse", "period"]) <= supply,
                sum_by(ship, ["customer", "period"]) >= demand,
            ],
        )
        prob.solve()

        self.assertEqual(prob.status, cp.OPTIMAL)

        # Export results
        result_df = variable_to_dataframe(ship)
        self.assertEqual(len(result_df), 8)

        # Verify supply constraints
        for _, row in supply_df.iterrows():
            w, t, s = row["warehouse"], row["period"], row["supply"]
            shipped = result_df[(result_df["warehouse"] == w) & (result_df["period"] == t)][
                "value"
            ].sum()
            self.assertLessEqual(shipped, s + 1e-6)

        # Verify demand constraints
        for _, row in demand_df.iterrows():
            c, t, d = row["customer"], row["period"], row["demand"]
            shipped = result_df[(result_df["customer"] == c) & (result_df["period"] == t)][
                "value"
            ].sum()
            self.assertGreaterEqual(shipped, d - 1e-6)

    def test_3d_direct_from_long_format(self):
        """Test 3D problem loading data directly from long-format DataFrame."""
        # All data in one long-format DataFrame
        df = pd.DataFrame(
            {
                "warehouse": ["W1", "W1", "W1", "W1", "W2", "W2", "W2", "W2"],
                "customer": ["C1", "C1", "C2", "C2", "C1", "C1", "C2", "C2"],
                "period": ["T1", "T2", "T1", "T2", "T1", "T2", "T1", "T2"],
                "cost": [10, 12, 15, 14, 20, 22, 18, 19],
            }
        )

        # Create index and parameter directly from 3D data
        shipments = set_from_dataframe(df, ["warehouse", "customer", "period"], name="shipments")
        cost = parameter_from_dataframe(
            df, ["warehouse", "customer", "period"], "cost", index=shipments
        )

        # Create variable
        ship = Variable(shipments, nonneg=True)

        # Simple optimization
        prob = cp.Problem(cp.Minimize(cost @ ship), [cp.sum(ship) >= 10])
        prob.solve()

        self.assertEqual(prob.status, cp.OPTIMAL)
        self.assertGreaterEqual(cp.sum(ship).value, 10 - 1e-6)


class TestSetFromSeries(unittest.TestCase):
    """Tests for set_from_series."""

    def test_basic(self):
        """Test creating Set from Series with duplicates."""
        s = pd.Series(["A", "B", "A", "C", "B"])
        result = set_from_series(s, name="items")
        self.assertEqual(len(result), 3)
        self.assertIn("A", result)
        self.assertIn("B", result)
        self.assertIn("C", result)

    def test_uses_series_name(self):
        """Test that Series name is used as Set name."""
        s = pd.Series(["X", "Y"], name="letters")
        result = set_from_series(s)
        self.assertEqual(result.name, "letters")


class TestSetFromIndex(unittest.TestCase):
    """Tests for set_from_index."""

    def test_simple_index(self):
        """Test creating Set from simple DataFrame index."""
        df = pd.DataFrame({"value": [1, 2, 3]}, index=["A", "B", "C"])
        df.index.name = "items"
        result = set_from_index(df)
        self.assertEqual(list(result), ["A", "B", "C"])
        self.assertEqual(result.name, "items")

    def test_multiindex(self):
        """Test creating Set from MultiIndex."""
        idx = pd.MultiIndex.from_tuples(
            [("W1", "C1"), ("W1", "C2"), ("W2", "C1")],
            names=["warehouse", "customer"],
        )
        df = pd.DataFrame({"value": [1, 2, 3]}, index=idx)
        result = set_from_index(df)
        self.assertTrue(result._is_compound)
        self.assertEqual(result.names, ("warehouse", "customer"))
        self.assertIn(("W1", "C1"), result)

    def test_3d_multiindex(self):
        """Test creating Set from 3-level MultiIndex."""
        idx = pd.MultiIndex.from_tuples(
            [
                ("W1", "C1", "T1"),
                ("W1", "C1", "T2"),
                ("W1", "C2", "T1"),
                ("W2", "C1", "T1"),
            ],
            names=["warehouse", "customer", "period"],
        )
        df = pd.DataFrame({"value": [1, 2, 3, 4]}, index=idx)
        result = set_from_index(df)
        self.assertTrue(result._is_compound)
        self.assertEqual(result.names, ("warehouse", "customer", "period"))
        self.assertEqual(len(result), 4)


class TestParameterFromSeries(unittest.TestCase):
    """Tests for parameter_from_series."""

    def test_simple_series(self):
        """Test creating Parameter from simple Series."""
        s = pd.Series({"W1": 100, "W2": 150}, name="supply")
        param = parameter_from_series(s)
        self.assertEqual(param.get_value("W1"), 100)
        self.assertEqual(param.get_value("W2"), 150)

    def test_multiindex_series(self):
        """Test creating Parameter from MultiIndex Series."""
        idx = pd.MultiIndex.from_tuples(
            [("W1", "C1"), ("W1", "C2"), ("W2", "C1")],
            names=["warehouse", "customer"],
        )
        s = pd.Series([10, 15, 20], index=idx, name="cost")
        param = parameter_from_series(s)
        self.assertEqual(param.get_value(("W1", "C1")), 10)
        self.assertEqual(param.get_value(("W2", "C1")), 20)

    def test_3d_multiindex_series(self):
        """Test creating Parameter from 3-level MultiIndex Series."""
        idx = pd.MultiIndex.from_tuples(
            [
                ("W1", "C1", "T1"),
                ("W1", "C1", "T2"),
                ("W1", "C2", "T1"),
                ("W2", "C1", "T1"),
            ],
            names=["warehouse", "customer", "period"],
        )
        s = pd.Series([10, 12, 15, 20], index=idx, name="cost")
        param = parameter_from_series(s)
        self.assertEqual(param.get_value(("W1", "C1", "T1")), 10)
        self.assertEqual(param.get_value(("W1", "C1", "T2")), 12)
        self.assertEqual(param.get_value(("W2", "C1", "T1")), 20)


if __name__ == "__main__":
    unittest.main()
