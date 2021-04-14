Internal DataFrame representation
=================================

Modin paritions data to scale efficiently.
To keep track of everything a few key classes are introduced: Frame, Partition, AxisPartiton and FrameManager.

* `Frame` is the class conforming to DataFrame Algebra.
* `Partition` is an element of a NxM grid which, when combined, represents the `Frame`
* `AxisPartition` is a joined group of `Parition`-s along either rows or labels column
* `PartitionManager` is the manager that implements the primitives used for DataFrame Algebra operations over `Partition`-s

.. toctree::
    :hidden:

    data
    partition
    axis_partition
    partition_manager
