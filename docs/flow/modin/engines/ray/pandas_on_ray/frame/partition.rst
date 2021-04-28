PandasOnRayFramePartition
"""""""""""""""""""""""""

The class is specific implementation of ``BaseFramePartition``, providing an API
to perform an operation on a block partition, namely, pandas DataFrame, using Ray as an execution engine.

In addition to wrapping a pandas DataFrame, the class also holds the following metadata:

* ``length`` - length of pandas DataFrame wrapped
* ``width`` - width of pandas DataFrame wrapped
* ``ip`` - node IP address that holds pandas DataFrame wrapped

An operation on a block partition can be performed in two modes:

* asyncronously - via :meth:`~modin.engines.ray.pandas_on_ray.frame.PandasOnRayFramePartition.apply`
* lazily - via :meth:`~modin.engines.ray.pandas_on_ray.frame.PandasOnRayFramePartition.add_to_apply_calls`

Public API
----------

.. autoclass:: modin.engines.ray.pandas_on_ray.frame.partition.PandasOnRayFramePartition
  :noindex:
  :members: