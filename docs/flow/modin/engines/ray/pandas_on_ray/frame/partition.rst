PandasOnRayFramePartition
"""""""""""""""""""""""""

The class is the specific implementation of :py:class:`~modin.engines.base.frame.partition.PandasFramePartition`,
providing the API to perform operations on a block partition, namely, ``pandas.DataFrame``, using Ray as an execution engine.

In addition to wrapping a pandas DataFrame, the class also holds the following metadata:

* ``length`` - length of pandas DataFrame wrapped
* ``width`` - width of pandas DataFrame wrapped
* ``ip`` - node IP address that holds pandas DataFrame wrapped

An operation on a block partition can be performed in two modes:

* asynchronously_ - via :meth:`~modin.engines.ray.pandas_on_ray.frame.partition.PandasOnRayFramePartition.apply`
* lazily_ - via :meth:`~modin.engines.ray.pandas_on_ray.frame.partition.PandasOnRayFramePartition.add_to_apply_calls`

Public API
----------

.. autoclass:: modin.engines.ray.pandas_on_ray.frame.partition.PandasOnRayFramePartition
  :members:

.. _asynchronously: https://en.wikipedia.org/wiki/Asynchrony_(computer_programming)
.. _lazily: https://en.wikipedia.org/wiki/Lazy_evaluation
