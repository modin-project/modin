PandasDataframeAxisPartition
""""""""""""""""""""""""""""

The class is base for any axis partition class of ``pandas`` storage format.

Subclasses must implement ``list_of_blocks`` which represents data wrapped by the ``PandasDataframePartition``
objects and creates something interpretable as a ``pandas.DataFrame``.

See ``modin.core.execution.ray.implementations.pandas_on_ray.partitioning.axis_partition.PandasOnRayDataframeAxisPartition``
for an example on how to override/use this class when the implementation needs to be augmented.

The ``PandasDataframeAxisPartition`` object has an invariant that requires that this
object is never returned from a function. It assumes that there will always be
``PandasDataframeAxisPartition`` object stored and structures itself accordingly.

Public API
----------

.. autoclass:: modin.core.dataframe.pandas.partitioning.axis_partition.PandasDataframeAxisPartition
  :members:
