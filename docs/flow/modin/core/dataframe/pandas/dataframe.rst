PandasDataframe
"""""""""""""""

:py:class:`~modin.core.dataframe.pandas.dataframe.dataframe.PandasDataframe` is a direct descendant of :py:class:`~modin.core.dataframe.base.dataframe.dataframe.ModinDataframe`. Its purpose is to implement the abstract interfaces for usage with all ``pandas``-based :doc:`storage formats</flow/modin/core/storage_formats/index>`.
:py:class:`~modin.core.dataframe.pandas.dataframe.dataframe.PandasDataframe` could be inherited and augmented further by any specific implementation which needs it to take special care of some behavior or to improve performance for certain execution engine.

The class serves as the intermediate level
between ``pandas`` query compiler and conforming partition manager. All queries formed
at the query compiler layer are ingested by this class and then conveyed jointly with the stored partitions
into the partition manager for processing. Direct partitions manipulation by this class is prohibited except
cases if an operation is strictly private or protected and called inside of the class only. The class provides
significantly reduced set of operations that fit plenty of pandas operations.

Main tasks of :py:class:`~modin.core.dataframe.pandas.dataframe.dataframe.PandasDataframe` are storage of partitions, manipulation with labels of axes and
providing set of methods to perform operations on the internal data.

As mentioned above, ``PandasDataframe`` shouldn't work with stored partitions directly and
the responsibility for modifying partitions array has to lay on :doc:`partitioning/partition_manager`. For example, method
:meth:`~modin.core.dataframe.pandas.dataframe.dataframe.PandasDataframe.broadcast_apply_full_axis` redirects applying
function to :meth:`~PandasDataframePartitionManager.broadcast_axis_partitions` method.

``Modin PandasDataframe`` can be created from ``pandas.DataFrame``, ``pyarrow.Table``
(methods :meth:`~modin.core.dataframe.pandas.dataframe.dataframe.PandasDataframe.from_pandas`,
:meth:`~modin.core.dataframe.pandas.dataframe.dataframe.PandasDataframe.from_arrow` are used respectively). Also,
``PandasDataframe`` can be converted to ``np.array``, ``pandas.DataFrame``
(methods :meth:`~modin.core.dataframe.pandas.dataframe.dataframe.PandasDataframe.to_numpy`,
:meth:`~modin.core.dataframe.pandas.dataframe.dataframe.PandasDataframe.to_pandas` are used respectively).

Manipulation with labels of axes happens using internal methods for changing labels on the new,
adding prefixes/suffixes etc.

Public API
----------

.. autoclass:: modin.core.dataframe.pandas.dataframe.dataframe.PandasDataframe
  :members:
