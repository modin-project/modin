PandasOnRayDataframeAxisPartition
"""""""""""""""""""""""""""""""""

This class is the specific implementation of :py:class:`~modin.core.dataframe.pandas.partitioning.axis_partition.PandasDataframeAxisPartition`,
providing the API to perform operations on an axis partition, using Ray as an execution engine. The axis partition is
a wrapper over a list of block partitions that are stored in this class.

Public API
----------

.. autoclass:: modin.core.execution.ray.implementations.pandas_on_ray.partitioning.axis_partition.PandasOnRayDataframeAxisPartition
  :members:
