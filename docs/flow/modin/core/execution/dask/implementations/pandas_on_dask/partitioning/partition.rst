PandasOnDaskDataframePartition
""""""""""""""""""""""""""""""

The class is the specific implementation of :py:class:`~modin.core.dataframe.pandas.partitioning.partition.PandasDataframePartition`,
providing the API to perform operations on a block partition, namely, ``pandas.DataFrame``, using Dask as the execution engine.

In addition to wrapping a ``pandas.DataFrame``, the class also holds the following metadata:

* ``length`` - length of ``pandas.DataFrame`` wrapped
* ``width`` - width of ``pandas.DataFrame`` wrapped
* ``ip`` - node IP address that holds ``pandas.DataFrame`` wrapped

An operation on a block partition can be performed in two modes:

* asynchronously_ - via :meth:`~modin.core.execution.dask.implementations.pandas_on_dask.partitioning.PandasOnDaskDataframePartition.apply`
* lazily_ - via :meth:`~modin.core.execution.dask.implementations.pandas_on_dask.partitioning.PandasOnDaskDataframePartition.add_to_apply_calls`

Public API
----------

.. autoclass:: modin.core.execution.dask.implementations.pandas_on_dask.partitioning.PandasOnDaskDataframePartition
  :members:

  .. _asynchronously: https://en.wikipedia.org/wiki/Asynchrony_(computer_programming)
  .. _lazily: https://en.wikipedia.org/wiki/Lazy_evaluation
