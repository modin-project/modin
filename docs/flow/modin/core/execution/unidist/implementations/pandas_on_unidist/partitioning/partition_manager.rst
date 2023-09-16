PandasOnUnidistDataframePartitionManager
""""""""""""""""""""""""""""""""""""""""

This class is the specific implementation of :py:class:`~modin.core.execution.unidist.generic.partitioning.GenericUnidistDataframePartitionManager`
using Unidist distributed engine. This class is responsible for partition manipulation and applying a function to
block/row/column partitions.

Public API
----------

.. autoclass:: modin.core.execution.unidist.implementations.pandas_on_unidist.partitioning.PandasOnUnidistDataframePartitionManager
  :members:
