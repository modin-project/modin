PandasDataframePartition
""""""""""""""""""""""""

The class is base for any partition class of ``pandas`` storage format and serves as the last level
on which operations that were conveyed from the partition manager are being performed on an
individual block partition.

The class provides an API that has to be overridden by child classes in order to manipulate
on data and metadata they store.

The public API exposed by the children of this class is used in :py:class:`~modin.core.dataframe.pandas.partitioning.partition_manager.PandasDataframePartitionManager`.

The objects wrapped by the child classes are treated as immutable by ``PandasDataframePartitionManager`` subclasses
and no logic for updating inplace.

Public API
----------

.. autoclass:: modin.core.dataframe.pandas.partitioning.partition.PandasDataframePartition
  :members:
