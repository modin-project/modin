BasePandasFrame
"""""""""""""""

The class is base for any frame class of ``pandas`` backend and serves as the intermediate level
between ``pandas`` query compiler and conforming partition manager. All queries formed
at the query compiler layer are ingested by this class and then conveyed jointly with the stored partitions
into the partition manager for processing. Direct partitions manipulation by this class is prohibited except
cases if an operation is striclty private or protected and called inside of the class only. The class provides
significantly reduced set of operations that fit plenty of pandas operations.

Main task of ``BasePandasFrame`` is storage of partitions, manipulation with labels of axes and
providing public interface for partition manipulation.

``BasePandasFrame`` can be created from ``pandas.DataFrame``, ``pyarrow.Table`` or from labels of this base frame
(methods :meth:`~modin.engines.base.frame.data.BasePandasFrame.from_pandas`,
:meth:`~modin.engines.base.frame.data.BasePandasFrame.from_arrow`,
:meth:`~modin.engines.base.frame.data.BasePandasFrame.from_labels` are used respectively). Also,
``BasePandasFrame`` can be converted to ``np.array``, ``pandas.DataFrame`` or labels of this
base frame(methods :meth:`~modin.engines.base.frame.data.BasePandasFrame.to_numpy`,
:meth:`~modin.engines.base.frame.data.BasePandasFrame.to_pandas`,
:meth:`~modin.engines.base.frame.data.BasePandasFrame.to_labels` are used respectively)

Manipulation with labels of axes happens using internal methods for changing labels on the new, 
adding prefixes/suffixes etc.

As already mentioned, ``BasePandasFrame`` doesn't work with stored array of partitions directly. 
A responsibility for the modifying partitions list lies with :doc:`partition_manager`. ``BasePandasFrame``
provides several methods to apply function along partitions. For example, method
:meth:`~modin.engines.base.frame.data.BasePandasFrame.broadcast_apply_full_axis` redirects applying
function to ``BaseFrameManager.broadcast_axis_partitions`` method.


Public API
----------

.. autoclass:: modin.engines.base.frame.data.BasePandasFrame
  :members:
