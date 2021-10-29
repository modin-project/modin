Modin Pandas Dataframe Objects
==============================

* :doc:`PandasDataframe <dataframe>` is the class conforming to Dataframe Algebra.
* :doc:`PandasDataframePartition <partitioning/partition>` implements ``Partition`` interface holding ``pandas.DataFrame``.
* :doc:`PandasDataframeAxisPartition <partitioning/axis_partition>` is a joined group of ``PandasDataframePartition``-s along some axis (either rows or labels)
* :doc:`PandasDataframePartitionManager <partitioning/partition_manager>` is the manager that implements the primitives used for Dataframe Algebra operations over ``PandasDataframePartition``-s

.. toctree::
    :hidden:

    dataframe
    partitioning/partition
    partitioning/axis_partition
    partitioning/partition_manager