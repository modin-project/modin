PandasOnPython Frame Objects
============================

This page describes implementation of :doc:`Base Frame Objects </flow/modin/engines/base/frame/index>`
specific for ``PandasOnPython`` backend. Since Python engine doesn't allow computation parallelization,
operations on partitions are performed sequentially. The absence of parallelization doesn't give any
perfomance speed-up, so ``PandasOnPython`` is used for testing purposes only.

* :doc:`Frame <data>`
* :doc:`Partition <partition>`
* :doc:`AxisPartition <axis_partition>`
* :doc:`PartitionManager <partition_manager>`

.. toctree::
    :hidden:

    data
    partition
    axis_partition
    partition_manager