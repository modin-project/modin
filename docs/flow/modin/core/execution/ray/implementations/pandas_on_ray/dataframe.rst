PandasOnRayDataframe
""""""""""""""""""""

The class is specific implementation of :py:class:`~modin.core.dataframe.pandas.dataframe.dataframe.PandasDataframe`
class using Ray distributed engine. It serves as an intermediate level between
:py:class:`~modin.core.storage_formats.pandas.query_compiler.PandasQueryCompiler` and
:py:class:`~modin.core.execution.ray.implementations.pandas_on_ray.partitioning.PandasOnRayDataframePartitionManager`.

Public API
----------

.. autoclass:: modin.core.execution.ray.implementations.pandas_on_ray.dataframe.PandasOnRayDataframe
  :members: