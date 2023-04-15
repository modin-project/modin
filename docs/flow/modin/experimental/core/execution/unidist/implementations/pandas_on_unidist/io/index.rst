:orphan:

IO module Description For Pandas-on-Unidist Execution
"""""""""""""""""""""""""""""""""""""""""""""""""""""

High-Level Module Overview
''''''''''''''''''''''''''

This module houses experimental functionality with pandas storage format and Unidist
engine. This functionality is concentrated in the :py:class:`~modin.experimental.core.execution.unidist.implementations.pandas_on_unidist.io.io.ExperimentalPandasOnUnidistIO`
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

The ``modin.experimental.core.execution.unidist.implementations.pandas_on_unidist`` module primarily houses utils and 
functions for the experimental IO class:

* ``io.py`` - submodule containing IO class and parse functions, which are responsible
  for data processing on the workers.

Public API
''''''''''

.. autoclass:: modin.experimental.core.execution.unidist.implementations.pandas_on_unidist.io.io.ExperimentalPandasOnUnidistIO
  :members:
