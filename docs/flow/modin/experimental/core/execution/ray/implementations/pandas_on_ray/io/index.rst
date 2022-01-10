:orphan:

IO module Description For Pandas-on-Ray Excecution
""""""""""""""""""""""""""""""""""""""""""""""""""

High-Level Module Overview
''''''''''''''''''''''''''

This module houses experimental functionality with pandas storage format and Ray
engine. This functionality is concentrated in the :py:class:`~modin.experimental.core.execution.ray.implementations.pandas_on_ray.io.io.ExperimentalPandasOnRayIO`
class, that contains methods, which extend typical pandas API to give user
more flexibility with IO operations.

Usage Guide
'''''''''''

In order to use the experimental features, just modify standard Modin import
statement as follows:

.. code-block:: python

  # import modin.pandas as pd
  import modin.experimental.pandas as pd

Submodules Description
''''''''''''''''''''''

``modin.experimental.core.execution.ray.implementations.pandas_on_ray`` module is used mostly for storing utils and 
functions for experimanetal IO class:

* ``io.py`` - submodule containing IO class and parse functions, which are responsible
  for data processing on the workers.

* ``sql.py`` - submodule with util functions for experimental ``read_sql`` function.

Public API
''''''''''

.. autoclass:: modin.experimental.core.execution.ray.implementations.pandas_on_ray.io.io.ExperimentalPandasOnRayIO
  :members:
