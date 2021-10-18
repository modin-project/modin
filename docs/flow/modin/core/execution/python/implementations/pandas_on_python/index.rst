PandasOnPython Dataframe implementation
=======================================

This page describes implementation of :doc:`base Dataframe Objects </flow/modin/core/dataframe/index>`
specific for `PandasOnPython` backend. Since Python engine doesn't allow computation parallelization,
operations on partitions are performed sequentially. The absence of parallelization doesn't give any
perfomance speed-up, so ``PandasOnPython`` is used for testing purposes only.

* :doc:`Dataframe <dataframe>`
* :doc:`Partition <partitioning/partition>`
* :doc:`AxisPartition <partitioning/axis_partition>`
* :doc:`PartitionManager <partitioning/partition_manager>`

.. toctree::
    :hidden:

    dataframe
    partitioning/partition
    partitioning/axis_partition
    partitioning/partition_manager