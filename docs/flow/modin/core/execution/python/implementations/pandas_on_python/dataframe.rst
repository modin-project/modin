PandasOnPythonDataframe
"""""""""""""""""""""""

The class is specific implementation of :py:class:`~modin.core.dataframe.pandas.dataframe.dataframe.PandasDataframe`
for `Python` execution engine. It serves as an intermediate level between
:py:class:`~modin.core.storage_formats.pandas.query_compiler.PandasQueryCompiler` and
:py:class:`~modin.core.execution.python.implementations.pandas_on_python.partitioning.partition_manager.PandasOnPythonDataframePartitionManager`.

Public API
----------

.. autoclass:: modin.core.execution.python.implementations.pandas_on_python.dataframe.dataframe.PandasOnPythonDataframe
  :members: