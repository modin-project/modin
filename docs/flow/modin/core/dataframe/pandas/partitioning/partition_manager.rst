PandasDataframePartitionManager
"""""""""""""""""""""""""""""""

The class is base for any partition manager class of ``pandas`` storage format and serves as
intermediate level between :doc:`Modin PandasDataframe <../dataframe>` and conforming :doc:`partition <partition>` class.
The class is responsible for partitions manipulation and applying a function to individual partitions:
block partitions, row partitions or column partitions, i.e. the class can form axis partitions from
block partitions to apply a function if an operation requires access to an entire column or row.
The class translates frame API into partition API and also can have some preprocessing operations
depending on the partition type for improving performance (for example,
:meth:`~modin.core.dataframe.pandas.partitioning.partition_manager.PandasDataframePartitionManager.preprocess_func`).

Main task of partition manager is to keep knowledge of how partitions are stored and managed
internal to itself, so surrounding code could use it via lean enough API without worrying about
implementation details.

Partition manager can apply user-passed (arbitrary) function in different modes:

* block-wise (apply a function to individual block partitions):

  * optionally accepting partition indices along each axis
  * optionally accepting an item to be split so parts of it would be sent to each partition

* along a full axis (apply a function to an entire column or row made up of block partitions when user function needs information about the whole axis)

It can also broadcast partitions from `right` to `left` when executing certain operations making
`right` partitions available for functions executed where `left` live.

..
  TODO: insert more text explaining "broadcast" term

Partition manager also is used to create "logical" partitions, or :doc:`axis partitions <axis_partition>`
by joining existing partitions along specified axis (either rows or labels),
and to concatenate different partition sets along given axis.

It also maintains mapping from "external" (end user-visible) indices along all axes to internal
indices which are actually pairs of indices of partitions and indices inside the partitions,
as well as manages conversion to numpy and pandas representations.


Public API
----------

.. autoclass:: modin.core.dataframe.pandas.partitioning.partition_manager.PandasDataframePartitionManager
  :members:
