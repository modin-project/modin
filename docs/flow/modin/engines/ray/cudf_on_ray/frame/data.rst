cuDFOnRayFrame
""""""""""""""

The class is the specific implementation of :py:class:`~modin.engines.base.frame.data.PandasFrame`
class using Ray distributed engine. It serves as an intermediate level between
:py:class:`~modin.backends.cudf.query_compiler.cuDFQueryCompiler` and
:py:class:`~modin.engines.ray.cudf_on_ray.frame.partition_manager.cuDFOnRayFramePartitionManager`.

Public API
----------

.. autoclass:: modin.engines.ray.cudf_on_ray.frame.data.cuDFOnRayFrame
  :members:
