Inspection tutorial
===================

This tutorial demonstrates the typical data inspection and risk-check pipeline.

Steps
-----

1. Load your dataset
2. Run `mlvern.data.inspect` pipeline
3. Review the generated report

Example
-------

.. code-block:: python

   import pandas as pd
   from mlvern.data.inspect import inspect_df

   df = pd.read_csv('data/sample.csv')
   report = inspect_df(df)
   report.save('reports/inspection.html')
