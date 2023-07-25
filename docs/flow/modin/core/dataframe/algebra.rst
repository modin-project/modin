:orphan:

Operators Module Description
""""""""""""""""""""""""""""

Brief description
'''''''''''''''''
Most of the functions that are evaluated by `QueryCompiler` can be categorized into
one of the patterns: Map, TreeReduce, Binary, Reduce, etc., called core operators. The ``modin.core.dataframe.algebra``
module provides templates to easily build such types of functions. These templates
are supposed to be used at the `QueryCompiler` level since each built function accepts
and returns `QueryCompiler`.

High-Level Module Overview
''''''''''''''''''''''''''
Each template class implements a
``register`` method, which takes functions to apply and
instantiate the related template. Functions that are passed to ``register`` will be executed
against converted to pandas and preprocessed in a template-specific way partition, so the function
would take one of the pandas object: ``pandas.DataFrame``, ``pandas.Series`` or ``pandas.DataFrameGroupbyObject``.

.. note:: 
    Currently, functions that are built in that way are supported only in a pandas
    storage format (i.e. can be used only in `PandasQueryCompiler`).

Algebra module provides templates for this type of function:

Map operator
-------------
Uniformly apply a function argument to each partition in parallel. 
**Note**: map function should not change the shape of the partitions.

.. figure:: /img/map_evaluation.svg
    :align: center

.. autoclass:: modin.core.dataframe.algebra.map.Map
  :members: register, apply

Reduce operator
---------------
Applies an argument function that reduces each column or row on the specified axis into a scalar, but requires knowledge about the whole axis.
Be aware that providing this knowledge may be expensive because the execution engine has to
concatenate partitions along the specified axis. Also, note that the execution engine expects
that the reduce function returns a one dimensional frame.

.. figure:: /img/reduce_evaluation.svg
    :align: center

.. autoclass:: modin.core.dataframe.algebra.reduce.Reduce
  :members: register, apply

TreeReduce operator
-------------------
Applies an argument function that reduces specified axis into a scalar. First applies map function to each partition
in parallel, then concatenates resulted partitions along the specified axis and applies reduce
function. In contrast with `Map function` template, here you're allowed to change partition shape
in the map phase. Note that the execution engine expects that the reduce function returns a one dimensional frame.

.. autoclass:: modin.core.dataframe.algebra.tree_reduce.TreeReduce
  :members: register, apply

Binary operator
---------------
Applies an argument function, that takes exactly two operands (first is always `QueryCompiler`).
If both operands are query compilers then the execution engine broadcasts partitions of
the right operand to the left.

.. figure:: /img/binary_evaluation.svg
    :align: center

.. warning::
    To be able to do frame broadcasting, partitioning along the index axis of both frames
    has to be equal, otherwise they need to be aligned first. The execution engine will do
    it automatically but note that this requires repartitioning, which is a much 
    more expensive operation than the binary function itself.

.. autoclass:: modin.core.dataframe.algebra.binary.Binary
  :members: register, apply

Fold operator
-------------
Applies an argument function that requires knowledge of the whole axis. Be aware that providing this knowledge may be
expensive because the execution engine has to concatenate partitions along the specified axis.

.. autoclass:: modin.core.dataframe.algebra.fold.Fold
  :members: register, apply

GroupBy operator
----------------
Evaluates GroupBy aggregation for that type of functions that can be executed via TreeReduce approach.
To be able to form groups engine broadcasts ``by`` partitions to each partition of the source frame.

.. autoclass:: modin.core.dataframe.algebra.groupby.GroupByReduce
  :members: register, apply

Default-to-pandas operator
--------------------------
Do :doc:`fallback to pandas </supported_apis/defaulting_to_pandas>` for passed function.


How to use UDFs with these operators
''''''''''''''''''''''''''''''''''''
Let's examine an example of how to use the algebra module to create your own
new functions.

Imagine you have a complex aggregation that can be implemented into a single query but
doesn't have any implementation in pandas API. If you know how to implement this
aggregation efficiently in a distributed frame, you may want to use one of the above described
patterns (e.g. ``TreeReduce``).

Let's implement a function that counts non-NA values for each column or row
(``pandas.DataFrame.count``). First, we need to determine the function type.
TreeReduce approach would be great: in a map phase, we'll count non-NA cells in each
partition in parallel and then just sum its results in the reduce phase.

To execute a TreeReduce function that does `count` + `sum` you can simply use the operator's ``.apply(...)``
method that takes and outputs a :py:class:`~modin.pandas.dataframe.DataFrame`:

.. code-block:: python

    from modin.core.dataframe.algebra import TreeReduce

    res_df = TreeReduce.apply(
        df,
        map_func=lambda df: df.count(),
        reduce_function=lambda df: df.sum()
    )

If you're going to use your custom-defined function quite often you may want
to wrap it into a separate function or assign it as a DataFrame's method:

.. code-block:: python

    import modin.pandas as pd

    def count_func(self):
        return TreeReduce.apply(
            self,
            map_func=lambda df: df.count(),
            reduce_function=lambda df: df.sum()
        )

    # you can then use the function as is
    res = count_func(df)

    # or assign it to the DataFrame's class and use it as a method
    pd.DataFrame.count_custom = count_func
    res = df.count_custom()

Many of the `pandas` API functions can be easily implemented this way, so if you find
out that one of your favorite function is still defaulted to pandas and decide to
contribute to Modin to add its implementation, you may use this example as a reference.
