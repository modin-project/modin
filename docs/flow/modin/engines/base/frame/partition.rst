PandasFramePartition
""""""""""""""""""""

The class is base for any partition class of ``pandas`` backend and serves as the last level
on which operations that were conveyed from the partition manager are being performed on an
individual block partition.

The class provides an API that has to be overridden by child classes in order to manipulate
on data and metadata they store.

The public API exposed by the children of this class is used in ``PandasFramePartitionManager``.

The objects wrapped by the child classes are treated as immutable by ``PandasFramePartitionManager`` subclasses
and no logic for updating inplace.

Public API
----------

.. autoclass:: modin.engines.base.frame.partition.PandasFramePartition
  :members:
