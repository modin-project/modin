:orphan:

Generic Unidist-based members
=============================

Objects which are storage format agnostic but require specific Unidist implementation
are placed in ``modin.core.execution.unidist.generic``.

Their purpose is to implement certain parallel I/O operations and to serve
as a foundation for building storage format specific objects:

* :py:class:`~modin.core.execution.unidist.generic.io.UnidistIO` -- implements parallel :meth:`~modin.core.execution.unidist.generic.io.UnidistIO.to_csv` and :meth:`~modin.core.execution.unidist.generic.io.UnidistIO.to_sql`.
* :py:class:`~modin.core.execution.unidist.generic.partitioning.GenericUnidistDataframePartitionManager` -- implements parallel :meth:`~modin.core.execution.unidist.generic.partitioning.GenericUnidistDataframePartitionManager.to_numpy`.

.. autoclass:: modin.core.execution.unidist.generic.partitioning.GenericUnidistDataframePartitionManager
  :members:
