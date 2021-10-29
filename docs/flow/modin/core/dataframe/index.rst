:orphan:

Base Modin Dataframe Objects
============================

Modin paritions data to scale efficiently.
To keep track of everything a few key classes are introduced: ``Dataframe``, ``Partition``, ``AxisPartiton`` and ``PartitionManager``.

* `Dataframe` is the class conforming to Dataframe Algebra.
* `Partition` is an element of a NxM grid which, when combined, represents the ``Dataframe``
* `AxisPartition` is a joined group of ``Partition``-s along some axis (either rows or columns)
* `PartitionManager` is the manager that implements the primitives used for Dataframe Algebra operations over ``Partition``-s

Each :doc:`storage format </flow/modin/core/storage_formats/index>` may have its own implementations of these Dataframe's entities.
Current stable implementations are the following:

* :doc:`Base Modin Dataframe <base/index>` defines a common interface and algebra operators for `Dataframe` implementations.
* :doc:`Pandas Dataframe <pandas/index>` is an implementation for any frame class of :doc:`pandas storage format </flow/modin/core/storage_formats/pandas/index>`.

.. note::
    At the current stage of Modin development, the base interfaces of the Dataframe objects are not defined yet.
    So for now the origin of all changes in the Dataframe interfaces is the :doc:`Dataframe for pandas storage format<pandas/index>`.

.. toctree::
    :hidden:

    base/index
    pandas/index
