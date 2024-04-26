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

This operator performs best when the number of partitions equals to the number of CPUs
so that each single partition gets processed in parallel. When the number of partitions is 1.5x greater than
the number of CPUs, Modin applies a heuristic to join some partitions to get "ideal" partitioning so that
each new partition gets processed in parallel.

Reduce operator
---------------
Applies an argument function that reduces each column or row on the specified axis into a scalar, but requires knowledge about the whole axis.
Be aware that providing this knowledge may be expensive because the execution engine has to
concatenate partitions along the specified axis. Also, note that the execution engine expects
that the reduce function returns a one dimensional frame.

.. figure:: /img/reduce_evaluation.svg
    :align: center

This operator performs best when the number of partitions (row or column partitions in depend on the specified axis)
equals to the number of CPUs so that each single axis partition gets processed in parallel.

TreeReduce operator
-------------------
Applies an argument function that reduces specified axis into a scalar. First applies map function to each partition
in parallel, then concatenates resulted partitions along the specified axis and applies reduce
function. In contrast with `Map function` template, here you're allowed to change partition shape
in the map phase. Note that the execution engine expects that the reduce function returns a one dimensional frame.

This operator performs best when the number of partitions (including the initial and intermediate stages)
equals to the number of CPUs so that each single axis partition gets processed in parallel.

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

This operator performs best when both operands have identical partitioning and the number of partitions of an operand
equals to the number of CPUs so that each single partition gets processed in parallel.

Fold operator
-------------
Applies an argument function that requires knowledge of the whole axis. Be aware that providing this knowledge may be
expensive because the execution engine has to concatenate partitions along the specified axis.

This operator performs best when the number of partitions (row or column partitions in depend on the specified axis)
equals to the number of CPUs so that each single axis partition gets processed in parallel.

GroupBy operator
----------------
Evaluates GroupBy aggregation for that type of functions that can be executed via TreeReduce approach.
To be able to form groups engine broadcasts ``by`` partitions to each partition of the source frame.

This operator performs best when the cardinality of ``by`` columns is low (small number of output groups).
At the ``Map`` stage, the operator computes the aggregation for each row partition individually, meaning,
that the ``Reduce`` stage takes a dataframe with the following number of rows:
``num_groups * n_row_parts``. If the number of groups is too high, there's a risk of getting a dataframe
with even bigger than the initial shape at the ``Reduce`` stage.

Default-to-pandas operator
--------------------------
Do :doc:`fallback to pandas </supported_apis/defaulting_to_pandas>` for passed function.

This operator has a performance penalty for going from a partitioned Modin DataFrame to pandas because of
the communication cost and single-threaded nature of pandas.


How to register your own function
'''''''''''''''''''''''''''''''''
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

To define the TreeReduce function that does `count` + `sum` we just need to register the
appropriate functions and then assign the result to the picked `QueryCompiler`
(`PandasQueryCompiler` in our case):

.. code-block:: python

    from modin.core.storage_formats import PandasQueryCompiler
    from modin.core.dataframe.algebra import TreeReduce

    PandasQueryCompiler.custom_count = TreeReduce.register(pandas.DataFrame.count, pandas.DataFrame.sum)

Then, we want to handle it from the :py:class:`~modin.pandas.dataframe.DataFrame`, so we need to create a way to do that:

.. code-block:: python

    import modin.pandas as pd

    def count_func(self, **kwargs):
        # The constructor allows you to pass in a query compiler as a keyword argument
        return self.__constructor__(query_compiler=self._query_compiler.custom_count(**kwargs))

    pd.DataFrame.count_custom = count_func

And then you can use it like you usually would:

.. code-block:: python

    df.count_custom(axis=1)

Many of the `pandas` API functions can be easily implemented this way, so if you find
out that one of your favorite function is still defaulted to pandas and decide to
contribute to Modin to add its implementation, you may use this example as a reference.
