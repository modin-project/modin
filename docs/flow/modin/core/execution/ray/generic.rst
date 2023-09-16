:orphan:

Generic Ray-based members
=========================

Objects which are storage format agnostic but require specific Ray implementation
are placed in ``modin.core.execution.ray.generic``.

Their purpose is to implement certain parallel I/O operations and to serve
as a foundation for building storage format specific objects:

.. autoclass:: modin.core.execution.ray.generic.io.RayIO
  :members:

.. autoclass:: modin.core.execution.ray.generic.partitioning.GenericRayDataframePartitionManager
  :members:
