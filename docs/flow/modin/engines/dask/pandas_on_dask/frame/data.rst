PandasOnDaskFrame
"""""""""""""""""

The class is the specific implementation of the dataframe algebra for the ``PandasOnDask`` backend.
It serves as an intermediate level between ``pandas`` query compiler and
:py:class:`~modin.engines.dask.pandas_on_dask.frame.partition_manager.PandasOnDaskFramePartitionManager`.

Public API
----------

.. autoclass:: modin.engines.dask.pandas_on_dask.frame.data.PandasOnDaskFrame
  :members:
