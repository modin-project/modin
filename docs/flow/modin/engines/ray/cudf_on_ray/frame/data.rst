cuDFOnRayFrame
""""""""""""""""

The class is specific implementation of :py:class:`~modin.engines.base.frame.data.BasePandasFrame`
class using Ray distributed engine. It serves an intermediate level between 
:py:class:`~modin.backends.cudf.query_compiler.cuDFQueryCompiler` and
:py:class:`~modin.engines.ray.cudf_on_ray.frame.partition_manager.cuDFOnRayFrameManager`.

Public API
----------

.. autoclass:: modin.engines.ray.cudf_on_ray.frame.data.cuDFOnRayFrame
  :members: