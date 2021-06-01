PandasOnRayFrameAxisPartition
"""""""""""""""""""""""""""""

This class is the specific implementation of :py:class:`~modin.engines.base.frame.axis_partition.PandasFrameAxisPartition`,
providing the API to perform operations on an axis partition, using Ray as an execution engine. The axis partition is
a wrapper over a list of block partitions that are stored in this class.

Public API
----------

.. autoclass:: modin.engines.ray.pandas_on_ray.frame.axis_partition.PandasOnRayFrameAxisPartition
  :members:

PandasOnRayFrameColumnPartition
"""""""""""""""""""""""""""""""

Public API
----------

.. autoclass:: modin.engines.ray.pandas_on_ray.frame.axis_partition.PandasOnRayFrameColumnPartition
  :members:

PandasOnRayFrameRowPartition
""""""""""""""""""""""""""""

Public API
----------

.. autoclass:: modin.engines.ray.pandas_on_ray.frame.axis_partition.PandasOnRayFrameRowPartition
  :members:
