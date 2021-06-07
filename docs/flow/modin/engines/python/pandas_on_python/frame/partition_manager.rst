PythonFrameManager
""""""""""""""""""

The class is specific implementation of :py:class:`~modin.engines.base.frame.partition_manager.PandasFramePartitionManager`
using Python as the execution engine. This class is responsible for partitions manipulation and applying
a funcion to block/row/column partitions.

Public API
----------

.. autoclass:: modin.engines.python.pandas_on_python.frame.partition_manager.PandasOnPythonFramePartitionManager
  :members: