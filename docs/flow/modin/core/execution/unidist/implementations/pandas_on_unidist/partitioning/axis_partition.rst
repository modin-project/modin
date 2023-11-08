PandasOnUnidistDataframeVirtualPartition
""""""""""""""""""""""""""""""""""""""""

This class is the specific implementation of :py:class:`~modin.core.dataframe.pandas.partitioning.axis_partition.PandasDataframeAxisPartition`,
providing the API to perform operations on an axis partition, using Unidist as an execution engine. The virtual partition is
a wrapper over a list of block partitions, which are stored in this class, with the capability to combine the smaller partitions into the one "virtual".

Public API
----------

.. autoclass:: modin.core.execution.unidist.implementations.pandas_on_unidist.partitioning.PandasOnUnidistDataframeVirtualPartition
  :members:

PandasOnUnidistDataframeColumnPartition
"""""""""""""""""""""""""""""""""""""""

Public API
----------

.. autoclass:: modin.core.execution.unidist.implementations.pandas_on_unidist.partitioning.PandasOnUnidistDataframeColumnPartition
  :members:

PandasOnUnidistDataframeRowPartition
""""""""""""""""""""""""""""""""""""

Public API
----------

.. autoclass:: modin.core.execution.unidist.implementations.pandas_on_unidist.partitioning.PandasOnUnidistDataframeRowPartition
  :members:
