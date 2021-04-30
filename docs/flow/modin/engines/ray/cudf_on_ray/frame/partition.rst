cuDFOnRayFramePartition
"""""""""""""""""""""""""

The class is specific implementation of ``BaseFramePartition``, providing an API
to perform an operation on a block partition, namely, cudf.DataFrame, using Ray as an execution engine.

An operation on a block partition can be performed in two modes:

* asyncronously - via :meth:`~modin.engines.ray.cudf_on_ray.frame.cuDFOnRayFramePartition.apply`
* lazily - via :meth:`~modin.engines.ray.cudf_on_ray.frame.cuDFOnRayFramePartition.add_to_apply_calls`

Public API
----------

.. autoclass:: modin.engines.ray.cudf_on_ray.frame.partition.cuDFOnRayFramePartition
  :noindex:
  :members: