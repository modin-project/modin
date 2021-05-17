:orphan:

Pandas-on-Ray Module Description
""""""""""""""""""""""""""""""""

High-Level Module Overview
''''''''''''''''''''''''''

This module houses experimental functionality with pandas backend and Ray
engine. This functionality is concentrated in the ``ExperimentalPandasOnRayIO`` class,
that contains methods, which extend typical pandas API to give user more flexibility
with IO operations.

Usage Guide
'''''''''''

In order to use the experimental features, just modify standard Modin import
statement as follows:

.. code-block:: python

  # import modin.pandas as pd
  import modin.experimental.pandas as pd

Implemented Operations
''''''''''''''''''''''

For now ``ExperimentalPandasOnRayIO`` implements two methods - ``read_sql`` and
``read_csv_glob``. The first method allows the user to use typical
``pandas.read_sql`` function extended with `Spark-like parameters
<https://spark.apache.org/docs/2.0.0/api/R/read.jdbc.html>`_ such as
``partition_column``, ``lower_bound`` and ``upper_bound``. With these parameters,
the user will be able to specify how to partition the imported data. The second
implemented method allows to read multiple CSV files simultaneously when a
`Python Glob <https://docs.python.org/3/library/glob.html>`_ object is provided
as a parameter.

Submodules Description
''''''''''''''''''''''

``modin.experimental.engines.pandas_on_ray`` module is used mostly for storing utils and 
functions for experimanetal IO class:

* ``io_exp.py`` - submodule containing IO class and parse functions, which are responsible
  for data processing on the workers.

* ``sql.py`` - submodule with util functions for experimental ``read_sql`` function.
