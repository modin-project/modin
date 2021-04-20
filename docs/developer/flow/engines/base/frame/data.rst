BasePandasFrame
"""""""""""""""

The class is base for any frame class of ``pandas`` backend and serves as the intermediate level
between ``pandas`` query compiler and conforming partition manager. All queries formed
at the query compiler layer are ingested by this class and then conveyed jointly with the stored partitions
into the partition manager for processing. Direct partitions manipulation by this class is prohibited except
cases if an operation is striclty private or protected and called inside of the class only. The class provides
significantly reduced set of operations that fit plenty of pandas operations.

Main task of ``BasePandasFrame`` is storage of partitions, manipulation with labels of axes and
providing set of methods to perform function from query compiler layer on the internal data.

``BasePandasFrame`` doesn't work with stored array of partitions directly. 
``BasePandasFrame`` only provides some interface to perform function along internal data. A responsibility for
the modifying partitions list lies with :doc:`partition_manager`. For example, method
:meth:`~modin.engines.base.frame.data.BasePandasFrame.broadcast_apply_full_axis` redirects applying
function to ``BaseFrameManager.broadcast_axis_partitions`` method.

``BasePandasFrame`` can be created from ``pandas.DataFrame``, ``pyarrow.Table`` 
(methods :meth:`~modin.engines.base.frame.data.BasePandasFrame.from_pandas`,
:meth:`~modin.engines.base.frame.data.BasePandasFrame.from_arrow` are used respectively). Also,
``BasePandasFrame`` can be converted to ``np.array``, ``pandas.DataFrame``
(methods :meth:`~modin.engines.base.frame.data.BasePandasFrame.to_numpy`,
:meth:`~modin.engines.base.frame.data.BasePandasFrame.to_pandas` are used respectively).

Manipulation with labels of axes happens using internal methods for changing labels on the new, 
adding prefixes/suffixes etc.


Public API
----------

.. autoclass:: modin.engines.base.frame.data.BasePandasFrame
  :members:
