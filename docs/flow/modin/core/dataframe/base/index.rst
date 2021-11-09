Modin BaseDataframe Interface
=============================

* :doc:`BaseDataframeAxisPartition <partitioning/axis_partition>` is an interface for a Core Dataframe object, representing a joined group of partitions along some axis (either rows or labels).

.. note::
    Common interfaces for most of the Modin Dataframe objects are not defined yet. Currently, all of the implementations
    inherit :doc:`Dataframe implementation for pandas storage format</flow/modin/core/dataframe/pandas/index>`.

.. toctree::
    :hidden:

    partitioning/axis_partition
