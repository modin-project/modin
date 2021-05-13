PandasOnDaskFramePartition
""""""""""""""""""""""""""

The class is specific implementation of ``BaseFramePartition``, providing an API
to perform an operation on a block partition, namely, pandas DataFrame, using Dask as an execution engine.

In addition to wrapping a pandas DataFrame, the class also holds the following metadata:

* ``length`` - length of pandas DataFrame wrapped
* ``width`` - width of pandas DataFrame wrapped
* ``ip`` - node IP address that holds pandas DataFrame wrapped

An operation on a block partition can be performed in two modes:

* asyncronously_ - via :meth:`~modin.engines.dask.pandas_on_dask.frame.partition.PandasOnDaskFramePartition.apply`
* lazily_ - via :meth:`~modin.engines.dask.pandas_on_dask.frame.partition.PandasOnDaskFramePartition.add_to_apply_calls`

Public API
----------

.. autoclass:: modin.engines.dask.pandas_on_dask.frame.partition.PandasOnDaskFramePartition
  :noindex:
  :members:

  .. _asyncronously: https://en.wikipedia.org/wiki/Asynchrony_(computer_programming)
  .. _lazily: https://en.wikipedia.org/wiki/Lazy_evaluation
