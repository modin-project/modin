PandasOnUnidistDataframe
""""""""""""""""""""""""

The class is specific implementation of :py:class:`~modin.core.dataframe.pandas.dataframe.dataframe.PandasDataframe`
class using Unidist distributed engine. It serves as an intermediate level between
:py:class:`~modin.core.storage_formats.pandas.query_compiler.PandasQueryCompiler` and
:py:class:`~modin.core.execution.unidist.implementations.pandas_on_unidist.partitioning.PandasOnUnidistDataframePartitionManager`.

Public API
----------

.. autoclass:: modin.core.execution.unidist.implementations.pandas_on_unidist.dataframe.PandasOnUnidistDataframe
  :members: