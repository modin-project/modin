:orphan:

Generic Ray-based members
=========================

Objects which are backend-agnostic but require specific Ray implementation
are placed in ``modin.engines.ray.generic``.

Their purpose is to implement certain parallel I/O operations and to serve
as a foundation for building backend-specific objects:

* :py:class:`~modin.engines.ray.generic.io.RayIO` -- implements parallel :meth:`~modin.engines.ray.generic.io.RayIO.to_csv` and :meth:`~modin.engines.ray.generic.io.RayIO.to_sql`.
* :py:class:`~modin.engines.ray.generic.frame.partition_manager.GenericRayFramePartitionManager` -- implements parallel :meth:`~modin.engines.ray.generic.frame.partition_manager.GenericRayFramePartitionManager.to_numpy`.

.. autoclass:: modin.engines.ray.generic.io.RayIO
  :members:

.. autoclass:: modin.engines.ray.generic.frame.partition_manager.GenericRayFramePartitionManager
  :members:
