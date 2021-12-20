cuDFOnRayDataframePartition
"""""""""""""""""""""""""""

The class is the specific implementation of :py:class:`~modin.core.dataframe.pandas.partitioning.partition.PandasDataframePartition`,
providing the API to perform operations on a block partition, namely, ``cudf.DataFrame``,
using Ray as an execution engine.

An operation on a block partition can be performed asynchronously_ in two ways:

* :meth:`~modin.core.execution.ray.implementations.cudf_on_ray.partitioning.partition.cuDFOnRayDataframePartition.apply` returns ``ray.ObjectRef``
  with integer key of operation result from internal storage.
* :meth:`~modin.core.execution.ray.implementations.cudf_on_ray.partitioning.partition.cuDFOnRayDataframePartition.add_to_apply_calls` returns
  a new :py:class:`~modin.core.execution.ray.implementations.cudf_on_ray.partitioning.partition.cuDFOnRayDataframePartition` object that is based on result of operation.

Public API
----------

.. autoclass:: modin.core.execution.ray.implementations.cudf_on_ray.partitioning.partition.cuDFOnRayDataframePartition
  :members:

.. _asynchronously: https://en.wikipedia.org/wiki/Asynchrony_(computer_programming)
