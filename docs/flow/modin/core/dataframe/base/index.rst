Purpose
=======

The :py:class:`~modin.core.dataframe.base.dataframe.dataframe.ModinDataframe` serves the purpose of describing and defining the :doc:`Core Dataframe Algebra </flow/modin/core/dataframe/algebra>`.

It is the core construction element and serves as the client for the :doc:`Modin Query Compiler</flow/modin/core/storage_formats/base/query_compiler>`. Descendants that offer implementations execute the queries from the compiler by invoking functions over partitions via a partition manager.

The partitions and partition manager interfaces are currently implementation-specific, but may
be standardized in the future.

The :py:class:`~modin.core.dataframe.base.dataframe.dataframe.ModinDataframe` and axis partitions are the interfaces that must be implemented by any :doc:`execution backend</flow/modin/core/execution/dispatching>` in order for it to be plugged in to Modin.
These classes are mostly abstract, however very simple and generic enough methods like
:py:meth:`~modin.core.dataframe.base.partitioning.BaseDataframeAxisPartition.force_materialization` can be implemented at the base level because for now we do not expect them to differ in any implementation.

ModinDataframe Interface
========================

* :doc:`ModinDataframe <dataframe>` is an abstract class which represents the algebra operators a dataframe must expose.
* :doc:`BaseDataframeAxisPartition <partitioning/axis_partition>` is an abstract class, representing a joined group of partitions along some axis (either rows or labels).

.. toctree::
    :hidden:

    dataframe
    partitioning/axis_partition
