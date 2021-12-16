Purpose
=======

``BaseDataframe`` serves the purpose of describing and defining the :doc:`Core Dataframe Algebra </flow/modin/core/dataframe/algebra>`.

It is the core construction element which serves as the client for :doc:`Modin Query Compiler</flow/modin/core/storage_formats/base/query_compiler>` and which implementations are actually executing the queries from the compiler by invoking functions over partition(s).

To execute the queries, a typical implementation also itroduces partitions and
partition manager, interfaces for which we might consider standardising in the future.
For now they're totally implementation-specific.

Base dataframe and axis partitions are the interfaces that must be implemented by any :doc:`execution backend</flow/modin/core/execution/dispatching>` that wants to be plugged in Modin.
These classes are mostly abstract, however very simple and generic enough methods like
:py:meth:`~modin.core.dataframe.base.partitioning.BaseDataframeAxisPartition.force_materialization` can be implemented at the base level because for now we do not expect them to differ in any implementation.

Modin BaseDataframe Interface
=============================

* :doc:`BaseDataframeAxisPartition <partitioning/axis_partition>` is an abstract class, representing a joined group of partitions along some axis (either rows or labels).

.. note::
    Common interfaces for most of the Modin Dataframe objects are not defined yet. Currently, all of the implementations
    inherit :doc:`Dataframe implementation for pandas storage format</flow/modin/core/dataframe/pandas/index>`.

.. toctree::
    :hidden:

    partitioning/axis_partition
