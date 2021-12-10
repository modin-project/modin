:orphan:

Generic Ray-based members
=========================

Objects which are storage format agnostic but require specific Ray implementation
are placed in ``modin.core.execution.ray.generic``.

Their purpose is to implement certain parallel I/O operations and to serve
as a foundation for building storage format specific objects:

* :py:class:`~modin.core.execution.ray.generic.io.io.RayIO` -- implements parallel :meth:`~modin.core.execution.ray.generic.io.io.RayIO.to_csv` and :meth:`~modin.core.execution.ray.generic.io.io.RayIO.to_sql`.
* :py:class:`~modin.core.execution.ray.generic.partitioning.partition_manager.GenericRayDataframePartitionManager` -- implements parallel :meth:`~modin.core.execution.ray.generic.partitioning.partition_manager.GenericRayDataframePartitionManager.to_numpy`.

.. autoclass:: modin.core.execution.ray.generic.io.io.RayIO
  :members:

.. autoclass:: modin.core.execution.ray.generic.partitioning.partition_manager.GenericRayDataframePartitionManager
  :members:
