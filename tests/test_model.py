"""Tests for cvxpy_or.model module."""

import unittest

import cvxpy as cp

from cvxpy_or import Model, Set, sum_by


class TestModel(unittest.TestCase):
    """Tests for the Model class."""

    def test_model_creation(self):
        """Test creating a model."""
        m = Model(name='test')
        self.assertEqual(m.name, 'test')
        self.assertEqual(len(m.variables), 0)
        self.assertEqual(len(m.constraints), 0)

    def test_add_variable(self):
        """Test adding a variable to model."""
        m = Model()
        idx = Set(['a', 'b', 'c'], name='items')
        x = m.add_variable(idx, name='x', nonneg=True)

        self.assertEqual(len(m.variables), 1)
        self.assertIn('x', m.variables)
        self.assertIs(m.get_variable('x'), x)

    def test_add_parameter(self):
        """Test adding a parameter to model."""
        m = Model()
        idx = Set(['a', 'b', 'c'], name='items')
        p = m.add_parameter(idx, data={'a': 1, 'b': 2, 'c': 3}, name='cost')

        self.assertEqual(len(m.parameters), 1)
        self.assertIn('cost', m.parameters)
        self.assertIs(m.get_parameter('cost'), p)

    def test_add_constraint(self):
        """Test adding constraints."""
        m = Model()
        idx = Set(['a', 'b', 'c'], name='items')
        x = m.add_variable(idx, name='x', nonneg=True)

        m.add_constraint('bound', x <= 10)
        self.assertEqual(len(m.constraints), 1)
        self.assertIn('bound', m.constraints)

    def test_minimize(self):
        """Test setting minimize objective."""
        m = Model()
        idx = Set(['a', 'b', 'c'], name='items')
        x = m.add_variable(idx, name='x')

        m.minimize(cp.sum(x))
        self.assertIsNotNone(m.objective)

    def test_solve_simple(self):
        """Test solving a simple problem."""
        m = Model(name='simple')
        idx = Set(['a', 'b'], name='items')

        x = m.add_variable(idx, name='x', nonneg=True)
        cost = m.add_parameter(idx, data={'a': 1, 'b': 2}, name='cost')

        m.add_constraint('upper', x <= 10)
        m.minimize(cost @ x)

        status = m.solve()
        self.assertEqual(status, cp.OPTIMAL)
        self.assertIsNotNone(m.value)
        self.assertAlmostEqual(m.value, 0.0, places=4)  # x=0 is optimal

    def test_transportation_with_model(self):
        """Test a transportation problem using Model."""
        m = Model(name='transport')

        # Sets
        warehouses = Set(['W1', 'W2'], name='warehouses')
        customers = Set(['C1', 'C2', 'C3'], name='customers')
        routes = Set.cross(warehouses, customers, name='routes')

        # Parameters
        supply_data = {'W1': 100, 'W2': 100}
        demand_data = {'C1': 50, 'C2': 70, 'C3': 30}
        cost_data = {
            ('W1', 'C1'): 1, ('W1', 'C2'): 2, ('W1', 'C3'): 3,
            ('W2', 'C1'): 4, ('W2', 'C2'): 1, ('W2', 'C3'): 2,
        }

        supply = m.add_parameter(warehouses, data=supply_data, name='supply')
        demand = m.add_parameter(customers, data=demand_data, name='demand')
        cost = m.add_parameter(routes, data=cost_data, name='cost')

        # Variable
        ship = m.add_variable(routes, nonneg=True, name='ship')

        # Constraints
        m.add_constraint('supply', sum_by(ship, 'warehouses') <= supply)
        m.add_constraint('demand', sum_by(ship, 'customers') >= demand)

        # Objective
        m.minimize(cost @ ship)

        # Solve
        status = m.solve()

        self.assertEqual(status, cp.OPTIMAL)
        self.assertIsNotNone(m.value)
        self.assertGreater(m.value, 0)

        # Check total demand is met
        total_shipped = sum(ship.value)
        self.assertGreaterEqual(total_shipped, 150 - 0.1)

    def test_summary(self):
        """Test model summary."""
        m = Model(name='test_model')
        idx = Set(['a', 'b'], name='items')
        m.add_variable(idx, name='x')
        m.add_constraint('bound', m.variables['x'] <= 10)

        summary = m.summary()
        self.assertIn('test_model', summary)
        self.assertIn('x', summary)
        self.assertIn('bound', summary)

    def test_get_variable_not_found(self):
        """Test error when variable not found."""
        m = Model()
        with self.assertRaises(KeyError):
            m.get_variable('nonexistent')

    def test_maximize(self):
        """Test maximize objective."""
        m = Model()
        idx = Set(['a', 'b'], name='items')
        x = m.add_variable(idx, name='x', nonneg=True)

        m.add_constraint('upper', x <= 5)
        m.maximize(cp.sum(x))

        status = m.solve()
        self.assertEqual(status, cp.OPTIMAL)
        self.assertAlmostEqual(m.value, 10.0, places=4)  # x=[5,5]


if __name__ == '__main__':
    unittest.main()
