cuDFOnRayDataframe
""""""""""""""""""

The class is the specific implementation of :py:class:`~modin.core.dataframe.pandas.dataframe.dataframe.PandasDataframe`
class using Ray distributed engine. It serves as an intermediate level between
:py:class:`~modin.core.storage_formats.cudf.query_compiler.cuDFQueryCompiler` and
:py:class:`~modin.core.execution.ray.implementations.cudf_on_ray.partitioning.cuDFOnRayDataframePartitionManager`.

Public API
----------

.. autoclass:: modin.core.execution.ray.implementations.cudf_on_ray.dataframe.cuDFOnRayDataframe
  :members:
