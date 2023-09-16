PandasOnRayDataframeVirtualPartition
""""""""""""""""""""""""""""""""""""

This class is the specific implementation of :py:class:`~modin.core.dataframe.pandas.partitioning.axis_partition.PandasDataframeAxisPartition`,
providing the API to perform operations on an axis partition, using Ray as an execution engine. The virtual partition is
a wrapper over a list of block partitions, which are stored in this class, with the capability to combine the smaller partitions into the one "virtual".

Public API
----------

.. autoclass:: modin.core.execution.ray.implementations.pandas_on_ray.partitioning.PandasOnRayDataframeVirtualPartition
  :members:

PandasOnRayDataframeColumnPartition
"""""""""""""""""""""""""""""""""""

Public API
----------

.. autoclass:: modin.core.execution.ray.implementations.pandas_on_ray.partitioning.PandasOnRayDataframeColumnPartition
  :members:

PandasOnRayDataframeRowPartition
""""""""""""""""""""""""""""""""

Public API
----------

.. autoclass:: modin.core.execution.ray.implementations.pandas_on_ray.partitioning.PandasOnRayDataframeRowPartition
  :members:
