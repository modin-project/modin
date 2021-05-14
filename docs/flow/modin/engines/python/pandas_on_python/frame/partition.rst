PandasOnPythonFramePartition
""""""""""""""""""""""""""""

The class is specific implementation of :py:class:`~modin.engines.base.frame.partition.BaseFramePartition`,
providing the API to perform operations on a block partition using Python as the execution engine.

In addition to wrapping a ``pandas.DataFrame``, the class also holds the following metadata:

* ``length`` - length of ``pandas.DataFrame`` wrapped
* ``width`` - width of ``pandas.DataFrame`` wrapped

An operation on a block partition can be performed in two modes:

* immediately via :meth:`~modin.engines.python.pandas_on_python.frame.partition.PandasOnPythonFramePartition.apply` - 
  in this case accumulated call queue and new function will be executed
  immediately.
* lazily_ via :meth:`~modin.engines.python.pandas_on_python.frame.partition.PandasOnPythonFramePartition.add_to_apply_calls` -
  in this case function will be added to the call queue and no computations
  will be done at the moment.

Public API
----------

.. autoclass:: modin.engines.python.pandas_on_python.frame.partition.PandasOnPythonFramePartition
  :members:

  .. _lazily: https://en.wikipedia.org/wiki/Lazy_evaluation