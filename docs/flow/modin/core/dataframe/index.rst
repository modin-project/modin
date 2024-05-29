:orphan:

Core Modin Dataframe Objects
============================

Modin partitions data to scale efficiently.
To keep track of everything a few key classes are introduced: ``Dataframe``, ``Partition``, ``AxisPartiton`` and ``PartitionManager``.

* ``Dataframe`` is the class conforming to Dataframe Algebra.
* ``Partition`` is an element of a NxM grid which, when combined, represents the ``Dataframe``
* ``AxisPartition`` is a joined group of ``Partition``-s along some axis (either rows or columns)
* ``PartitionManager`` is the manager that implements the primitives used for Dataframe Algebra operations over ``Partition``-s

Each :doc:`storage format </flow/modin/core/storage_formats/index>`, execution engine, and each execution system (storage format + execution engine)
may have its own implementations of these Core Dataframe's entities.
Current stable implementations are the following:

* :doc:`Base ModinDataframe <base/index>` defines a common interface and algebra operators for `Dataframe` implementations.

Storage format specific:

* :doc:`Modin PandasDataframe <pandas/index>` is an implementation for any frame class of :doc:`pandas storage format </flow/modin/core/storage_formats/pandas/index>`.

Engine specific:

* :doc:`Modin GenericRayDataframe </flow/modin/core/execution/ray/generic>` is an implementation for any frame class that works on Ray execution engine.
* :doc:`Modin GenericUnidistDataframe </flow/modin/core/execution/unidist/generic>` is an implementation for any frame class that works on Unidist execution engine.

Execution system specific:

* :doc:`Modin PandasOnRayDataframe </flow/modin/core/execution/ray/implementations/pandas_on_ray/index>` is a specialization of the Core Modin Dataframe for ``PandasOnRay`` execution.
* :doc:`Modin PandasOnDaskDataframe </flow/modin/core/execution/dask/implementations/pandas_on_dask/index>` is specialization of the Core Modin Dataframe for ``PandasOnDask`` execution.
* :doc:`Modin PandasOnPythonDataframe </flow/modin/core/execution/python/implementations/pandas_on_python/index>` is a specialization of the Core Modin Dataframe for ``PandasOnPython`` execution.
* :doc:`Modin PandasOnUnidistDataframe </flow/modin/core/execution/unidist/implementations/pandas_on_unidist/index>` is a specialization of the Core Modin Dataframe for ``PandasOnUnidist`` execution.

.. note::
    At the current stage of Modin development, the base interfaces of the Dataframe objects are not defined yet.
    So for now the origin of all changes in the Dataframe interfaces is the :doc:`Dataframe for pandas storage format<pandas/index>`.

.. toctree::
    :hidden:

    base/index
    pandas/index
