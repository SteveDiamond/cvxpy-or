"""Tests for xarray I/O functionality."""

from __future__ import annotations

import unittest

import cvxpy as cp
import numpy as np
import xarray as xr

from cvxpy_or import (
    Parameter,
    Set,
    Variable,
    parameter_from_dataarray,
    parameter_to_dataarray,
    set_from_dataarray,
    sum_by,
    variable_like_dataarray,
    variable_to_dataarray,
)


class TestSetFromDataArray(unittest.TestCase):
    """Tests for set_from_dataarray."""

    def test_1d_dataarray(self):
        """Test creating Set from 1D DataArray."""
        da = xr.DataArray(
            [10, 20, 30],
            dims=["warehouse"],
            coords={"warehouse": ["W1", "W2", "W3"]},
        )
        s = set_from_dataarray(da)
        self.assertEqual(list(s), ["W1", "W2", "W3"])
        self.assertFalse(s._is_compound)
        self.assertEqual(s.name, "warehouse")

    def test_1d_with_name(self):
        """Test creating Set from 1D DataArray with custom name."""
        da = xr.DataArray(
            [10, 20],
            dims=["warehouse"],
            coords={"warehouse": ["W1", "W2"]},
        )
        s = set_from_dataarray(da, name="origins")
        self.assertEqual(s.name, "origins")

    def test_2d_dataarray(self):
        """Test creating Set from 2D DataArray."""
        da = xr.DataArray(
            [[10, 20], [30, 40]],
            dims=["warehouse", "customer"],
            coords={"warehouse": ["W1", "W2"], "customer": ["C1", "C2"]},
        )
        s = set_from_dataarray(da)
        self.assertTrue(s._is_compound)
        self.assertEqual(s.names, ("warehouse", "customer"))
        self.assertEqual(len(s), 4)
        self.assertIn(("W1", "C1"), s)
        self.assertIn(("W2", "C2"), s)

    def test_3d_dataarray(self):
        """Test creating Set from 3D DataArray."""
        da = xr.DataArray(
            np.zeros((2, 2, 2)),
            dims=["warehouse", "customer", "period"],
            coords={
                "warehouse": ["W1", "W2"],
                "customer": ["C1", "C2"],
                "period": ["P1", "P2"],
            },
        )
        s = set_from_dataarray(da)
        self.assertEqual(len(s), 8)
        self.assertEqual(s.names, ("warehouse", "customer", "period"))
        self.assertIn(("W1", "C1", "P1"), s)
        self.assertIn(("W2", "C2", "P2"), s)

    def test_empty_dims_raises(self):
        """Test that empty DataArray raises error."""
        da = xr.DataArray(42)  # scalar, no dims
        with self.assertRaises(ValueError):
            set_from_dataarray(da)


class TestParameterFromDataArray(unittest.TestCase):
    """Tests for parameter_from_dataarray."""

    def test_1d_parameter(self):
        """Test creating Parameter from 1D DataArray."""
        da = xr.DataArray(
            [100, 80, 120],
            dims=["warehouse"],
            coords={"warehouse": ["Seattle", "Denver", "Chicago"]},
        )
        param = parameter_from_dataarray(da, name="supply")
        self.assertEqual(param.shape, (3,))
        self.assertEqual(param.get_value("Seattle"), 100)
        self.assertEqual(param.get_value("Denver"), 80)
        self.assertEqual(param.get_value("Chicago"), 120)

    def test_2d_parameter(self):
        """Test creating Parameter from 2D DataArray."""
        da = xr.DataArray(
            [[10, 20], [30, 40]],
            dims=["warehouse", "customer"],
            coords={"warehouse": ["W1", "W2"], "customer": ["C1", "C2"]},
        )
        param = parameter_from_dataarray(da, name="cost")
        self.assertEqual(param.shape, (4,))
        # Check values match C-order flattening
        self.assertEqual(param.get_value(("W1", "C1")), 10)
        self.assertEqual(param.get_value(("W1", "C2")), 20)
        self.assertEqual(param.get_value(("W2", "C1")), 30)
        self.assertEqual(param.get_value(("W2", "C2")), 40)

    def test_with_existing_index(self):
        """Test creating Parameter with existing Set index."""
        warehouses = Set(["W1", "W2"], name="warehouses")
        customers = Set(["C1", "C2"], name="customers")
        routes = Set.cross(warehouses, customers)

        da = xr.DataArray(
            [[10, 20], [30, 40]],
            dims=["warehouses", "customers"],
            coords={"warehouses": ["W1", "W2"], "customers": ["C1", "C2"]},
        )
        param = parameter_from_dataarray(da, index=routes, name="cost")
        self.assertIs(param.index, routes)


class TestVariableLikeDataArray(unittest.TestCase):
    """Tests for variable_like_dataarray."""

    def test_variable_creation(self):
        """Test creating Variable from DataArray template."""
        da = xr.DataArray(
            [[0, 0], [0, 0]],
            dims=["warehouse", "customer"],
            coords={"warehouse": ["W1", "W2"], "customer": ["C1", "C2"]},
        )
        var = variable_like_dataarray(da, name="ship", nonneg=True)
        self.assertEqual(var.shape, (4,))
        self.assertTrue(var.is_nonneg())

    def test_variable_1d(self):
        """Test creating 1D Variable from DataArray template."""
        da = xr.DataArray([0, 0], dims=["warehouse"], coords={"warehouse": ["W1", "W2"]})
        var = variable_like_dataarray(da, name="x")
        self.assertEqual(var.shape, (2,))
        self.assertEqual(var.index.name, "x")


class TestVariableToDataArray(unittest.TestCase):
    """Tests for variable_to_dataarray."""

    def test_simple_variable_export(self):
        """Test exporting 1D Variable to DataArray."""
        warehouses = Set(["W1", "W2", "W3"], name="warehouse")
        var = Variable(warehouses, nonneg=True)

        # Solve simple problem to get values
        prob = cp.Problem(cp.Minimize(cp.sum(var)), [var >= [1, 2, 3]])
        prob.solve()

        da = variable_to_dataarray(var)
        self.assertEqual(da.dims, ("warehouse",))
        self.assertEqual(list(da.coords["warehouse"].values), ["W1", "W2", "W3"])
        np.testing.assert_array_almost_equal(da.values, [1, 2, 3])

    def test_compound_variable_export(self):
        """Test exporting 2D Variable to DataArray."""
        warehouses = Set(["W1", "W2"], name="warehouse")
        customers = Set(["C1", "C2"], name="customer")
        routes = Set.cross(warehouses, customers)

        var = Variable(routes, nonneg=True)

        # Solve to get values
        prob = cp.Problem(cp.Minimize(cp.sum(var)), [var >= [1, 2, 3, 4]])
        prob.solve()

        da = variable_to_dataarray(var)
        self.assertEqual(da.dims, ("warehouse", "customer"))
        self.assertEqual(list(da.coords["warehouse"].values), ["W1", "W2"])
        self.assertEqual(list(da.coords["customer"].values), ["C1", "C2"])
        self.assertEqual(da.shape, (2, 2))
        # Check specific values
        self.assertAlmostEqual(float(da.sel(warehouse="W1", customer="C1")), 1)
        self.assertAlmostEqual(float(da.sel(warehouse="W2", customer="C2")), 4)

    def test_unsolved_variable_raises(self):
        """Test that exporting unsolved Variable raises error."""
        warehouses = Set(["W1", "W2"], name="warehouse")
        var = Variable(warehouses)
        with self.assertRaises(ValueError):
            variable_to_dataarray(var)


class TestParameterToDataArray(unittest.TestCase):
    """Tests for parameter_to_dataarray."""

    def test_parameter_export(self):
        """Test exporting Parameter to DataArray."""
        warehouses = Set(["W1", "W2"], name="warehouse")
        param = Parameter(warehouses, data={"W1": 100, "W2": 200})

        da = parameter_to_dataarray(param)
        self.assertEqual(da.dims, ("warehouse",))
        np.testing.assert_array_almost_equal(da.values, [100, 200])


class TestRoundTrip(unittest.TestCase):
    """Tests for round-trip conversion."""

    def test_parameter_roundtrip(self):
        """Test DataArray -> Parameter -> DataArray."""
        original = xr.DataArray(
            [[10, 20], [30, 40]],
            dims=["warehouse", "customer"],
            coords={"warehouse": ["W1", "W2"], "customer": ["C1", "C2"]},
            name="cost",
        )
        param = parameter_from_dataarray(original)
        result = parameter_to_dataarray(param)

        xr.testing.assert_equal(original, result)

    def test_variable_roundtrip(self):
        """Test Variable -> solve -> DataArray with same shape."""
        template = xr.DataArray(
            [[0, 0], [0, 0]],
            dims=["warehouse", "customer"],
            coords={"warehouse": ["W1", "W2"], "customer": ["C1", "C2"]},
        )
        var = variable_like_dataarray(template, name="ship", nonneg=True)

        # Solve simple problem
        prob = cp.Problem(cp.Minimize(cp.sum(var)), [var >= 1])
        prob.solve()

        result = variable_to_dataarray(var)
        self.assertEqual(result.dims, ("warehouse", "customer"))
        self.assertEqual(result.shape, (2, 2))
        self.assertEqual(list(result.coords["warehouse"].values), ["W1", "W2"])


class TestXArrayTransportationProblem(unittest.TestCase):
    """End-to-end test with xarray workflow."""

    def test_full_workflow(self):
        """Test complete xarray-based transportation problem."""
        # Define data as DataArrays
        cost = xr.DataArray(
            [[2.5, 1.0], [2.0, 1.5], [1.0, 2.5]],
            dims=["warehouse", "customer"],
            coords={
                "warehouse": ["Seattle", "Denver", "Chicago"],
                "customer": ["NYC", "LA"],
            },
        )

        supply = xr.DataArray(
            [100, 80, 120],
            dims=["warehouse"],
            coords={"warehouse": ["Seattle", "Denver", "Chicago"]},
        )

        demand = xr.DataArray(
            [150, 100], dims=["customer"], coords={"customer": ["NYC", "LA"]}
        )

        # Create model objects
        cost_param = parameter_from_dataarray(cost, name="cost")
        supply_param = parameter_from_dataarray(supply, name="supply")
        demand_param = parameter_from_dataarray(demand, name="demand")
        ship = variable_like_dataarray(cost, name="ship", nonneg=True)

        # Build problem using sum_by (function, not method)
        prob = cp.Problem(
            cp.Minimize(cost_param @ ship),
            [
                sum_by(ship, "warehouse") <= supply_param,
                sum_by(ship, "customer") >= demand_param,
            ],
        )
        prob.solve()

        self.assertEqual(prob.status, cp.OPTIMAL)

        # Export result as DataArray
        result = variable_to_dataarray(ship)
        self.assertEqual(result.dims, ("warehouse", "customer"))

        # Verify constraints are satisfied
        shipped_per_warehouse = result.sum(dim="customer")
        shipped_per_customer = result.sum(dim="warehouse")

        for w in ["Seattle", "Denver", "Chicago"]:
            self.assertLessEqual(
                float(shipped_per_warehouse.sel(warehouse=w)),
                float(supply.sel(warehouse=w)) + 1e-6,
            )

        for c in ["NYC", "LA"]:
            self.assertGreaterEqual(
                float(shipped_per_customer.sel(customer=c)),
                float(demand.sel(customer=c)) - 1e-6,
            )


if __name__ == "__main__":
    unittest.main()
