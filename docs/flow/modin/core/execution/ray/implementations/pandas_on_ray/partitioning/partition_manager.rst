PandasOnRayDataframePartitionManager
""""""""""""""""""""""""""""""""""""

This class is the specific implementation of :py:class:`~modin.core.execution.ray.generic.partitioning.partition_manager.GenericRayDataframePartitionManager`
using Ray distributed engine. This class is responsible for partition manipulation and applying a funcion to
block/row/column partitions.

Public API
----------

.. autoclass:: modin.core.execution.ray.implementations.pandas_on_ray.partitioning.partition_manager.PandasOnRayDataframePartitionManager
  :members:
