PandasOnPythonDataframePartition
""""""""""""""""""""""""""""""""

The class is specific implementation of :py:class:`~modin.core.dataframe.pandas.partitioning.partition_manager.PandasDataframePartitionManager`
using Python as the execution engine. This class is responsible for partitions manipulation and applying
a function to block/row/column partitions.

Public API
----------

.. autoclass:: modin.core.execution.python.implementations.pandas_on_python.partitioning.partition_manager.PandasOnPythonDataframePartitionManager
  :members: