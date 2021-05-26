Base Frame Objects
==================

Modin paritions data to scale efficiently.
To keep track of everything a few key classes are introduced: ``Frame``, ``Partition``, ``AxisPartiton`` and ``PartitionManager``.

* :doc:`Frame <data>` is the class conforming to DataFrame Algebra.
* :doc:`Partition <partition>` is an element of a NxM grid which, when combined, represents the ``Frame``
* :doc:`AxisPartition <axis_partition>` is a joined group of ``Parition``-s along some axis (either rows or labels)
* :doc:`PartitionManager <partition_manager>` is the manager that implements the primitives used for DataFrame Algebra operations over ``Partition``-s

.. toctree::
    :hidden:

    data
    partition
    axis_partition
    partition_manager
