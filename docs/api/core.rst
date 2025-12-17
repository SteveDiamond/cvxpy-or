Core Classes
============

The core classes provide set-based indexing for CVXPY.

Set
---

.. autoclass:: cvxpy_or.Set
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__, __len__, __contains__, __iter__, __or__, __and__, __sub__, __xor__

Variable
--------

.. autoclass:: cvxpy_or.Variable
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__, __getitem__

Parameter
---------

.. autoclass:: cvxpy_or.Parameter
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__, __getitem__

Model
-----

.. autoclass:: cvxpy_or.Model
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Filtering
---------

.. autofunction:: cvxpy_or.where
