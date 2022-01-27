ModinDataframe
""""""""""""""

The :py:class:`~modin.core.dataframe.base.dataframe.dataframe.ModinDataframe` is the parent class for all dataframes - regardless of what storage format they are backed by. Its purpose is to define the algebra operators that must be exposed by a dataframe.

This class exposes the dataframe algebra and is meant to be subclassed by all dataframe implementations.
Descendants of this class implement the algebra, and act as the intermediate level
between the query compiler and the underlying execution details (e.g. the conforming partition manager). The class provides
a significantly reduced set of operations that can be composed to form any pandas query.

The :py:class:`~modin.core.dataframe.pandas.dataframe.dataframe.PandasDataframe` is an example of a descendant of this class. It currently has implementations for some of the operators
exposed in this class, and is currently being refactored to include implementations for all of the algebra operators. Please
refer to the :doc:`PandasDataframe documentation </flow/modin/core/dataframe/pandas/dataframe>` for more information.

The :py:class:`~modin.core.dataframe.base.dataframe.dataframe.ModinDataframe` is independent of implementation specific details such as partitioning, storage format, or execution engine.

Public API
----------

.. autoclass:: modin.core.dataframe.base.dataframe.dataframe.ModinDataframe
  :members:
