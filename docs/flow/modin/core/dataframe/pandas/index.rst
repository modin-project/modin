Modin PandasDataframe Objects
=============================

``modin.core.dataframe.pandas`` is the package which houses common implementations
of different Modin internal classes used by most `pandas`-based :doc:`storage formats</flow/modin/core/storage_formats/index>`.

It also double-serves as the full example of how to implement Modin execution backend pieces (sans the :doc:`execution part</flow/modin/core/execution/dispatching>` which is absent here),
as it implements everything an execution backend needs to be fully conformant to Modin expectations.

* :doc:`PandasDataframe <dataframe>` is the class conforming to Dataframe Algebra.
* :doc:`PandasDataframePartition <partitioning/partition>` implements ``Partition`` interface holding ``pandas.DataFrame``.
* :doc:`PandasDataframeAxisPartition <partitioning/axis_partition>` is a joined group of ``PandasDataframePartition``-s along some axis (either rows or labels)
* :doc:`PandasDataframePartitionManager <partitioning/partition_manager>` is the manager that implements the primitives used for Dataframe Algebra operations over ``PandasDataframePartition``-s
* :doc:`ModinDtypes <metadata/dtypes>`
* :doc:`ModinIndex <metadata/index>`

.. toctree::
    :hidden:

    dataframe
    partitioning/partition
    partitioning/axis_partition
    partitioning/partition_manager
    metadata/dtypes
    metadata/index
