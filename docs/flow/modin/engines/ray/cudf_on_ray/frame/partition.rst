cuDFOnRayFramePartition
"""""""""""""""""""""""""

The class is specific implementation of :py:class:`~modin.engines.base.frame.partition.BaseFramePartition`,
providing the API to perform operations on a block partition, namely, ``cudf.DataFrame``,
using Ray as an execution engine.

An operation on a block partition can be performed by two ways:

* :meth:`~modin.engines.ray.cudf_on_ray.frame.partition.cuDFOnRayFramePartition.apply` returns ``ray.ObjectRef``
  with integer key of operation result from internal storage.
* :meth:`~modin.engines.ray.cudf_on_ray.frame.partition.cuDFOnRayFramePartition.add_to_apply_calls` returns 
  the new one ``cuDFOnRayFramePartition`` object that is based on result of operation.

Public API
----------

.. autoclass:: modin.engines.ray.cudf_on_ray.frame.partition.cuDFOnRayFramePartition
  :members: