Purpose
=======
``BaseDataframe`` serves the purpose of describing and defining the internal dataframe algebra.

It is the core construction element which serves as a middle layer between high-level pandas-imitating API
and lower-level query compiler. ``BaseDataframe`` is the interface which implementations actually translate the dataframe algebra calls to queries to the underlying compiler.

The purpose of such translation is to reduce the vast amount of public pandas API to something smaller but sufficient.
Query compiler level could reduce the API even further depending on the implementation.

Base dataframe and axis partitions are the interfaces that must be implemented by any backend that wants to be plugged in Modin.
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
