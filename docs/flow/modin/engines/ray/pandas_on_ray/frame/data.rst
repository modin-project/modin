PandasOnRayFrame
""""""""""""""""

The class is specific implementation of :py:class:`~modin.engines.base.frame.data.BasePandasFrame`
class using Ray distributed engine. It serves as an intermediate level between
:py:class:`~modin.backends.pandas.query_compiler.PandasQueryCompiler` and
:py:class:`~modin.engines.ray.pandas_on_ray.frame.partition_manager.PandasOnRayFrameManager`.

Public API
----------

.. autoclass:: modin.engines.ray.pandas_on_ray.frame.data.PandasOnRayFrame
  :members: