:orphan:

PandasOnPython Dataframe implementation
=======================================

This page describes implementation of :doc:`Modin PandasDataframe Objects </flow/modin/core/dataframe/pandas/index>`
specific for `PandasOnPython` execution. Since Python engine doesn't allow computation parallelization,
operations on partitions are performed sequentially. The absence of parallelization doesn't give any
perfomance speed-up, so ``PandasOnPython`` is used for testing purposes only.

* :doc:`PandasOnPythonDataframe <dataframe>`
* :doc:`PandasOnPythonDataframePartition <partitioning/partition>`
* :doc:`PandasOnPythonDataframeAxisPartition <partitioning/axis_partition>`
* :doc:`PandasOnPythonDataframePartitionManager <partitioning/partition_manager>`

.. toctree::
    :hidden:

    dataframe
    partitioning/partition
    partitioning/axis_partition
    partitioning/partition_manager