Function Module Description
"""""""""""""""""""""""""""

Brief description
'''''''''''''''''
Most of the functions that are evaluated by `QueryCompiler` can be categorized into
one of the patterns: Map, MapReduce, Binary functions, Fold functions, etc. Function
module provides templates to easily build such types of functions. These templates
are supposed to be used at the `QueryCompiler` level since each built function accepts
and returns `QueryCompiler`.

High-Level Module Overview
''''''''''''''''''''''''''
Each template represented with a class with the corresponding name and implements
``register`` method, which takes functions to apply in an appropriate way and
instantiate the related template. Functions that are passed to the ``register``
will be executed under deserialized and preprocessed (depends on the template)
partitions, so the function would take one of the pandas object: ``pandas.DataFrame``,
``pandas.Series`` or ``pandas.DataFrameGroupbyObject``.

.. note:: 
    Currently, functions that are built in that way are supported only in a pandas
    backend (can be used only in `PandasQueryCompiler`).

Function module provides templates for this type of function:

Map functions
-------------
Apply function to each partition in parallel. **Note**: map function should not change the shape of the partitions.

.. figure:: /img/map_evaluation.svg
    :align: center

Reduction functions
-------------------
Function that reduces specified axis into a scalar, but requires knowledge about the whole axis.
Be aware that providing this knowledge may be expensive because the execution engine has to
concatenate partitions along the specified axis. Also, note that the execution engine expects
that the reduction function returns a one dimensional frame.

.. figure:: /img/reduce_evaluation.svg
    :align: center

MapReduce Functions
-------------------
Function that reduces specified axis into a scalar. First applies map function to each partition
in parallel, then concatenates resulted partitions along the specified axis and applies reduction
function. In contrast with `Map function` template, here you're allowed to change partition shape
in the map phase. Note that the execution engine expects that the reduction function returns a one dimensional frame.

Binary functions
----------------
Function that takes two operands (left is always `QueryCompiler`) and evaluates function along them.
If both operands are query compilers then the execution engine broadcasts partitions of
the right operand to the left.

.. figure:: /img/binary_evaluation.svg
    :align: center

.. warning::
    To be able to do frame broadcasting, partitioning along the index axis of both frames
    has to be equal, otherwise they need to be aligned first. The execution engine will do
    it automatically but note that this requires repartitioning, which is a much 
    more expensive operation than the binary function itself.

Fold functions
--------------
Function that requires knowledge of the whole axis. Be aware that providing this knowledge may be
expensive because the execution engine has to concatenate partitions along the specified axis.

GroupBy functions
-----------------
Evaluates GroupBy aggregation for that type of functions that can be executed via MapReduce approach.
To be able to form groups engine broadcasts `by` partitions to each partition of the source frame.

Default-to-pandas functions
---------------------------
Do :ref:`fallback to pandas <defaulting-to-pandas-mechanism>` for passed function.


How to register your own function
'''''''''''''''''''''''''''''''''
Let's examine the actual example of how to use the function module in a sense of adding
new functions.

Imagine you have a complex aggregation that can be implemented into a single query but
doesn't have any implementation in pandas API. If you know how to implement this
aggregation efficiently in a distributed frame, you may want to use one of the described
patterns. 

Let's implement a function that counts non-NA values for each column or row
(`pandas.DataFrame.count`). First, we need to determine the function type.
MapReduce approach would be great: in a map phase, we'll count non-NA cells in each
partition in parallel and then just sum its results in the reduce phase.

To define the MapReduce function that does `count` + `sum` we just need to register the
appropriate functions and then assign the result to the picked `QueryCompiler`
(`PandasQueryCompiler` in our case):

.. code-block:: python

    from modin.backends import PandasQueryCompiler
    from modin.data_management.functions import MapReduceFunction

    PandasQueryCompiler.custom_count = MapReduceFunction.register(pandas.DataFrame.count, pandas.DataFrame.sum)

Then, we want to handle it from the DataFrame, so we need to create a way to do that:

.. code-block:: python

    import modin.pandas as pd

    def count_func(self, **kwargs):
        # The constructor allows you to pass in a query compiler as a keyword argument
        return self.__constructor__(query_compiler=self._query_compiler.custom_count(**kwargs))

    pd.DataFrame.count_custom = count_func

And then you can use it like you usually would:

.. code-block:: python

    df.count_custom(axis=1)

Much of the `pandas` API function can be easily implemented this way, so if you'll find
out that some of your favorite function is still defaulted to pandas and decide to
contribute to Modin to add its implementation, you may use this example as a reference.
