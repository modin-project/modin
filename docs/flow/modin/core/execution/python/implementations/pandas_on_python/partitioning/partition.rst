PandasOnPythonDataframePartition
""""""""""""""""""""""""""""""""

The class is specific implementation of :py:class:`~modin.core.dataframe.pandas.partitioning.partition.PandasDataframePartition`,
providing the API to perform operations on a block partition using Python as the execution engine.

In addition to wrapping a ``pandas.DataFrame``, the class also holds the following metadata:

* ``length`` - length of ``pandas.DataFrame`` wrapped
* ``width`` - width of ``pandas.DataFrame`` wrapped

An operation on a block partition can be performed in two modes:

* immediately via :meth:`~modin.core.execution.python.implementations.pandas_on_python.partitioning.partition.PandasOnPythonDataframePartition.apply` - 
  in this case accumulated call queue and new function will be executed
  immediately.
* lazily_ via :meth:`~modin.core.execution.python.implementations.pandas_on_python.partitioning.partition.PandasOnPythonDataframePartition.add_to_apply_calls` -
  in this case function will be added to the call queue and no computations
  will be done at the moment.

Public API
----------

.. autoclass:: modin.core.execution.python.implementations.pandas_on_python.partitioning.partition.PandasOnPythonDataframePartition
  :members:

  .. _lazily: https://en.wikipedia.org/wiki/Lazy_evaluation