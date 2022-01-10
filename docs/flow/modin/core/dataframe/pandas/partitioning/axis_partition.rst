PandasDataframeAxisPartition
""""""""""""""""""""""""""""

The class implements abstract interface methods from :py:class:`~modin.core.dataframe.base.partitioning.axis_partition.BaseDataframeAxisPartition`
giving the means for a sibling :doc:`partition manager<partition_manager>` to actually work with the axis-wide partitions.

The class is base for any axis partition class of ``pandas`` storage format.

Subclasses must implement ``list_of_blocks`` which represents data wrapped by the :py:class:`~modin.core.dataframe.pandas.partitioning.partition.PandasDataframePartition`
objects and creates something interpretable as a ``pandas.DataFrame``.

See :py:class:`~modin.core.execution.ray.implementations.pandas_on_ray.partitioning.axis_partition.PandasOnRayDataframeAxisPartition`
for an example on how to override/use this class when the implementation needs to be augmented.

The :py:class:`~modin.core.dataframe.pandas.partitioning.axis_partition.PandasDataframeAxisPartition` object has an invariant that requires that this
object is never returned from a function. It assumes that there will always be
``PandasDataframeAxisPartition`` object stored and structures itself accordingly.

Public API
----------

.. autoclass:: modin.core.dataframe.pandas.partitioning.axis_partition.PandasDataframeAxisPartition
  :members:
