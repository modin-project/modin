PandasOnDaskDataframePartitionManager
"""""""""""""""""""""""""""""""""""""

This class is the specific implementation of :py:class:`~modin.core.dataframe.pandas.partitioning.partition_manager.PandasDataframePartitionManager`
using Dask as the execution engine. This class is responsible for partition manipulation and applying a function to
block/row/column partitions.

Public API
----------

.. autoclass:: modin.core.execution.dask.implementations.pandas_on_dask.partitioning.PandasOnDaskDataframePartitionManager
  :members:
