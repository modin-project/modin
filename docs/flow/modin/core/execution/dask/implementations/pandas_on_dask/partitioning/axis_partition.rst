PandasOnDaskDataframeAxisPartition
""""""""""""""""""""""""""""""""""

The class is the specific implementation of :py:class:`~modin.core.dataframe.pandas.partitioning.axis_partition.PandasDataframeAxisPartition`,
providing the API to perform operations on an axis (column or row) partition using Dask as the execution engine.
The axis partition is a wrapper over a list of block partitions that are stored in this class.

Public API
----------

.. autoclass:: modin.core.execution.dask.implementations.pandas_on_dask.partitioning.axis_partition.PandasOnDaskDataframeAxisPartition
  :members:

PandasOnDaskDataframeColumnPartition
""""""""""""""""""""""""""""""""""""

Public API
----------

.. autoclass:: modin.core.execution.dask.implementations.pandas_on_dask.partitioning.axis_partition.PandasOnDaskDataframeColumnPartition
  :members:

PandasOnDaskDataframeRowPartition
"""""""""""""""""""""""""""""""""

Public API
----------

.. autoclass:: modin.core.execution.dask.implementations.pandas_on_dask.partitioning.axis_partition.PandasOnDaskDataframeRowPartition
  :members:
