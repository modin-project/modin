PandasOnDaskFrameAxisPartition
""""""""""""""""""""""""""""""

The class is the specific implementation of :py:class:`~modin.engines.base.frame.axis_partition.PandasFrameAxisPartition`,
providing the API to perform operations on an axis (column or row) partition using Dask as the execution engine.
The axis partition is a wrapper over a list of block partitions that are stored in this class.

Public API
----------

.. autoclass:: modin.engines.dask.pandas_on_dask.frame.axis_partition.PandasOnDaskFrameAxisPartition
  :members:

PandasOnDaskFrameColumnPartition
""""""""""""""""""""""""""""""""

Public API
----------

.. autoclass:: modin.engines.dask.pandas_on_dask.frame.axis_partition.PandasOnDaskFrameColumnPartition
  :members:

PandasOnDaskFrameRowPartition
"""""""""""""""""""""""""""""

Public API
----------

.. autoclass:: modin.engines.dask.pandas_on_dask.frame.axis_partition.PandasOnDaskFrameRowPartition
  :members:
