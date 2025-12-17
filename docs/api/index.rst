API Reference
=============

This section provides detailed API documentation for all cvxpy-or modules.

.. toctree::
   :maxdepth: 2

   core
   aggregations
   constraints
   io
   display
   validation

Quick Reference
---------------

**Core Classes**

- :class:`cvxpy_or.Set` - Ordered set of elements for indexing
- :class:`cvxpy_or.Variable` - CVXPY Variable indexed by a Set
- :class:`cvxpy_or.Parameter` - CVXPY Parameter indexed by a Set
- :class:`cvxpy_or.Model` - Problem builder and solver wrapper

**Aggregations**

- :func:`cvxpy_or.sum_by` - Sum over groups
- :func:`cvxpy_or.mean_by` - Mean over groups
- :func:`cvxpy_or.min_by` - Min over groups
- :func:`cvxpy_or.max_by` - Max over groups

**Constraints**

- :func:`cvxpy_or.at_most_k`, :func:`cvxpy_or.exactly_k`, :func:`cvxpy_or.at_least_k` - Cardinality
- :func:`cvxpy_or.implies`, :func:`cvxpy_or.mutex`, :func:`cvxpy_or.one_of` - Logical
- :func:`cvxpy_or.bounds`, :func:`cvxpy_or.flow_balance` - Bounds and flow

**pandas I/O**

- :func:`cvxpy_or.set_from_series`, :func:`cvxpy_or.set_from_dataframe` - Create Sets
- :func:`cvxpy_or.parameter_from_dataframe`, :func:`cvxpy_or.parameter_from_series` - Create Parameters
- :func:`cvxpy_or.variable_to_dataframe`, :func:`cvxpy_or.parameter_to_dataframe` - Export

**Display**

- :func:`cvxpy_or.print_variable`, :func:`cvxpy_or.print_parameter` - Print values
- :func:`cvxpy_or.print_solution` - Print solution summary

**Validation**

- :func:`cvxpy_or.validate_keys`, :func:`cvxpy_or.validate_numeric` - Validate data
- :class:`cvxpy_or.ValidationError` - Exception class
