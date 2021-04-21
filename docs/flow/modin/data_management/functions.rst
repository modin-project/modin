Function Module Description
"""""""""""""""""""""""""""

Brief description
'''''''''''''''''
Most of the functions that are evaluated by `QueryCompiler` can be categorized into one of the patterns: Map, MapReduce, Binary functions, Fold functions, etc. Function module provides templates to easily build such types of functions. These templates are supposed to be used at the `QueryCompiler` level since each built function accepts and returns `QueryCompiler`.

High-Level Module Overview
''''''''''''''''''''''''''
Function module provides templates for this type of function:

* Binary functions — Function that takes two operands (left is always `QueryCompiler`) and evaluates function along them. Engine efficiently broadcasts partitions of the right operand to the left if necessary. 
* Fold functions — Function that requires knowledge of the whole axis. Be aware that providing this knowledge may be expensive because the execution engine has to concatenate partitions along the specified axis.
* GroupBy functions — Evaluates GroupBy aggregation for that type of functions that can be executed via MapReduce approach. To be able to form groups engine efficiently broadcasts `by` partitions to each partition of the source frame.
* Map functions — Apply function to each partition in parallel. Note, that the map function should not change the shape of the partitions.
* MapReduce functions — Function that reduces specified axis into a scalar. First applies map function to each partition in parallel, then concatenates resulted partitions along the specified axis and applies reduction function. Note that the execution engine expects that the reduction function returns a scalar.
* Reduction functions — Function that reduces specified axis into a scalar, but requires knowledge about the whole axis. Be aware that providing this knowledge may be expensive because the execution engine has to concatenate partitions along the specified axis. Also, note that the execution engine expects that the reduction function returns a scalar.
* Default-to-pandas functions — Do fallback to pandas for passed function.

Each template represented with a class with the corresponding name and implements ``register`` method, which takes functions to apply in an appropriate way and instantiate the related template. Functions that are passed to the ``register`` will be executed under deserialized and preprocessed (depends on the template) partitions, so the function would take one of the pandas object: ``pandas.DataFrame``, ``pandas.Series`` or ``pandas.DataFrameGroupbyObject``.

.. note:: Currently, functions that are built in that way are supported only in a pandas backend (can be used only in `PandasQueryCompiler`).

How to register your own function
'''''''''''''''''''''''''''''''''
Let's examine the actual example of how to use the function module in a sense of adding new functions.

Imagine you have a complex aggregation that can be implemented into a single query but doesn't have any implementation in pandas API. If you know how to implement this aggregation efficiently in a distributed frame, you may want to use one of the described patterns. 

Let's implement a function that counts non-NA values for each column or row (`pandas.DataFrame.count`). First, we need to determine the function type. MapReduce approach would be great: in a map phase, we'll count non-NA cells in each partition in parallel and then just sum its results in the reduce phase.

To define the MapReduce function that does `count` + `sum` we just need to register the appropriate functions and then assign the result to the picked `QueryCompiler` (`PandasQueryCompiler` in our case): 
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

Much of the pandas API function can be easily implemented this way, so if you'll find out that some of your favorite function is still defaulted to pandas and decide to contribute to Modin to add its implementation, you may use this example as a reference.
