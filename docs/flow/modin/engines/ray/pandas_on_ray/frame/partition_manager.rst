PandasOnRayFramePartitionManager
""""""""""""""""""""""""""""""""

This class is the specific implementation of :py:class:`~modin.engines.base.frame.partition_manager.PandasFramePartitionManager`
using Ray distributed engine. This class is responsible for partition manipulation and applying a funcion to
block/row/column partitions.

Public API
----------

.. autoclass:: modin.engines.ray.pandas_on_ray.frame.partition_manager.PandasOnRayFramePartitionManager
  :members:
