PandasOnDaskDataframe
"""""""""""""""""""""

The class is the specific implementation of the dataframe algebra for the `Dask` execution engine.
It serves as an intermediate level between ``pandas`` query compiler and
:py:class:`~modin.core.execution.dask.implementations.pandas_on_dask.partitioning.PandasOnDaskDataframePartitionManager`.

Public API
----------

.. autoclass:: modin.core.execution.dask.implementations.pandas_on_dask.dataframe.PandasOnDaskDataframe
  :members:
