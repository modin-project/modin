PandasOnDaskFramePartitionManager
"""""""""""""""""""""""""""""""""

This class is the specific implementation of :py:class:`~modin.engines.base.frame.partition_manager.PandasFramePartitionManager`
using Dask as the execution engine. This class is responsible for partition manipulation and applying a funcion to
block/row/column partitions.

Public API
----------

.. autoclass:: modin.engines.dask.pandas_on_dask.frame.partition_manager.PandasOnDaskFramePartitionManager
  :members:
