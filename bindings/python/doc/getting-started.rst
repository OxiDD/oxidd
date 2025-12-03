Getting Started
===============

Installation
------------

Install OxiDD using pip:

.. code-block:: bash

   pip install oxidd
```

Usage
-----

The following is a very simple example of using OxiDD's BDDs:

.. code-block:: python
   :linenos:

   from oxidd.bdd import BDDManager

   # Create a manager for up to 100,000,000 nodes with an apply cache for
   # 1,000,000 entries and 1 worker thread
   manager = BDDManager(100_000_000, 1_000_000, 1)

   # Create 10 variables
   x = [manager.var(i) for i in manager.add_vars(10)]

   f = x[0] & x[1]
   g = x[0] & ~x[0]

   assert f.satisfiable()
   assert g.sat_count_float(10) == 0

   # Visualize the DD functions using OxiDD-vis: Open https://oxidd.net/vis in
   # your web browser (use Firefox or a Chromium-based browser), click on the
   # diagrams icon at the top left and then on the "Add diagram source" button.
   # Enter the URL printed when executing the following method (should be
   # http://localhost:4000). OxiDD-vis will then poll for DDs served via the
   # `visualize` and `visualize_with_names` methods. The DD with the label "My
   # first BDD" should now appear in the list. Click on it and enjoy the
   # visualization!
   manager.visualize_with_names("My first BDD", [(f, "f"), (g, "g")])
