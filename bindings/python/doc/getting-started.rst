Getting Started
===============

The following is a very simple example of using OxiDD's BDDs:

.. code-block:: python
   :linenos:

   from oxidd.bdd import BDDManager

   # Create a manager for up to 100,000,000 nodes with an apply cache for
   # 1,000,000 entries and 1 worker thread
   manager = BDDManager(100_000_000, 1_000_000, 1)

   # Create 10 variables
   x = [manager.new_var() for i in range(10)]

   assert (x[0] & x[1]).satisfiable()
   assert (x[0] & ~x[0]).sat_count_float(10) == 0
