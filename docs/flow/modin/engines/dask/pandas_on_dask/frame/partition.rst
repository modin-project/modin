PandasOnDaskFramePartition
""""""""""""""""""""""""""

The class is specific implementation of :py:class:`~modin.engines.base.frame.partition.BaseFramePartition`,
providing the API to perform operations on a block partition, namely, ``pandas.DataFrame``, using Dask as the execution engine.

In addition to wrapping a pandas DataFrame, the class also holds the following metadata:

* ``length`` - length of pandas DataFrame wrapped
* ``width`` - width of pandas DataFrame wrapped
* ``ip`` - node IP address that holds pandas DataFrame wrapped

An operation on a block partition can be performed in two modes:

* asynchronously_ - via :meth:`~modin.engines.dask.pandas_on_dask.frame.partition.PandasOnDaskFramePartition.apply`
* lazily_ - via :meth:`~modin.engines.dask.pandas_on_dask.frame.partition.PandasOnDaskFramePartition.add_to_apply_calls`

Public API
----------

.. autoclass:: modin.engines.dask.pandas_on_dask.frame.partition.PandasOnDaskFramePartition
  :members:

  .. _asynchronously: https://en.wikipedia.org/wiki/Asynchrony_(computer_programming)
  .. _lazily: https://en.wikipedia.org/wiki/Lazy_evaluation
