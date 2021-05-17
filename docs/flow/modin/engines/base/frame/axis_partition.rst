BaseFrameAxisPartition
""""""""""""""""""""""

The class is base for any axis partition class and serves as the last level on which
operations that were conveyed from the partition manager are being performed on an entire column or row.

The class provides an API that has to be overridden by the child classes in order to manipulate
on a list of block partitions (making up column or row partition) they store.

The procedures that use this class and its methods assume that they have some global knowledge
about the entire axis. This may require the implementation to use concatenation or append on the
list of block partitions.

The ``BaseFrameManager`` object that controls these objects (through the API exposed here) has an invariant
that requires that this object is never returned from a function. It assumes that there will always be
``BaseFramePartition`` object stored and structures itself accordingly.

Public API
----------

.. autoclass:: modin.engines.base.frame.axis_partition.BaseFrameAxisPartition
  :members:

PandasFrameAxisPartition
""""""""""""""""""""""""

The class is base for any axis partition class of ``pandas`` backend.

Subclasses must implement ``list_of_blocks`` which represents data wrapped by the ``BaseFramePartition``
objects and creates something interpretable as a pandas DataFrame.

See ``modin.engines.ray.pandas_on_ray.axis_partition.PandasOnRayFrameAxisPartition``
for an example on how to override/use this class when the implementation needs to be augmented.

Public API
----------

.. autoclass:: modin.engines.base.frame.axis_partition.PandasFrameAxisPartition
  :members:
