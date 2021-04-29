cuDFOnRayFrameManager
"""""""""""""""""""""

The class is specific implementation of ``RayFrameManager``. It serves as and intermediate level
between :doc:`cuDFOnRayFrame <data>` and :doc:`cuDFOnRayFramePartition <partition>` class.
This class is responsible for partitions manipulation and applying a funcion to
block/row/column partitions.

Public API
----------

.. .. autoclass:: modin.engines.ray.cudf_on_ray.frame.partition_manager.cuDFOnRayFrameManager
..   :noindex:
..   :members: