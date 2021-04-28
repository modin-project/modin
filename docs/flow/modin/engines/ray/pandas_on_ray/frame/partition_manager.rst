PandasOnRayFrameManager
"""""""""""""""""""""""""""

The class is specific implementation of ``BaseFrameManager`` uusing Ray distributed engine.
This class is responsible for partitions manipulation and applying a funcion to
block/row/column partitions.

Public API
----------

.. autoclass:: modin.engines.ray.pandas_on_ray.frame.partition_manager.PandasOnRayFrameManager
  :members: