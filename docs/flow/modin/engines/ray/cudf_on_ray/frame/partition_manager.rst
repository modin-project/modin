cuDFOnRayFramePartitionManager
""""""""""""""""""""""""""""""

This class is the specific implementation of :py:class:`~modin.engines.ray.generic.frame.partition_manager.GenericRayFramePartitionManager`.
It serves as an intermediate level between :py:class:`~modin.engines.ray.cudf_on_ray.frame.data.cuDFOnRayFrame`
and :py:class:`~modin.engines.ray.cudf_on_ray.frame.partition.cuDFOnRayFramePartition` class.
This class is responsible for partition manipulation and applying a function to
block/row/column partitions.

Public API
----------

.. autoclass:: modin.engines.ray.cudf_on_ray.frame.partition_manager.cuDFOnRayFramePartitionManager
  :members:
