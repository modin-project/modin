BaseQueryCompiler
"""""""""""""""""

Brief description
'''''''''''''''''
:py:class:`~modin.core.storage_formats.base.query_compiler.BaseQueryCompiler` is an abstract class of query compiler, and sets a common interface
that every other query compiler implementation in Modin must follow. The Base class contains a basic
implementations for most of the interface methods, all of which
:doc:`fallback to pandas </supported_apis/defaulting_to_pandas>`.

Subclassing :py:class:`~modin.core.storage_formats.base.query_compiler.BaseQueryCompiler`
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
If you want to add new type of query compiler to Modin the new class needs to inherit
from :py:class:`~modin.core.storage_formats.base.query_compiler.BaseQueryCompiler` and implement the abstract methods:

- :py:meth:`~modin.core.storage_formats.base.query_compiler.BaseQueryCompiler.from_pandas` build query compiler from pandas DataFrame.
- :py:meth:`~modin.core.storage_formats.base.query_compiler.BaseQueryCompiler.from_arrow` build query compiler from Arrow Table.
- :py:meth:`~modin.core.storage_formats.base.query_compiler.BaseQueryCompiler.to_pandas` get query compiler representation as pandas DataFrame.
- :py:meth:`~modin.core.storage_formats.base.query_compiler.BaseQueryCompiler.default_to_pandas` do :doc:`fallback to pandas </supported_apis/defaulting_to_pandas>` for the passed function. 
- :py:meth:`~modin.core.storage_formats.base.query_compiler.BaseQueryCompiler.finalize` finalize object constructing.
- :py:meth:`~modin.core.storage_formats.base.query_compiler.BaseQueryCompiler.free` trigger memory cleaning.

(Please refer to the code documentation to see the full documentation for these functions).

This is a minimum set of operations to ensure a new query compiler will function in the Modin architecture,
and the rest of the API can safely default to the pandas implementation via the base class implementation. 
To add a storage format specific implementation for some of the query compiler operations, just override 
the corresponding method in your query compiler class.

Example
'''''''
As an exercise let's define a new query compiler in `Modin`, just to see how easy it is.
Usually, the query compiler routes formed queries to the underlying :doc:`frame </flow/modin/core/dataframe/index>` class,
which submits operators to an execution engine. For the sake
of simplicity and independence of this example, our execution engine will be the pandas itself.

We need to inherit a new class from :py:class:`~modin.core.storage_formats.base.query_compiler.BaseQueryCompiler` and implement all of the abstract methods.
In this case, with `pandas` as an execution engine, it's trivial:

.. code-block:: python

    from modin.core.storage_formats import BaseQueryCompiler

    class DefaultToPandasQueryCompiler(BaseQueryCompiler):
        def __init__(self, pandas_df):
            self._pandas_df = pandas_df

        @classmethod
        def from_pandas(cls, df, *args, **kwargs):
            return cls(df)

        @classmethod
        def from_arrow(cls, at, *args, **kwargs):
            return cls(at.to_pandas())

        def to_pandas(self):
            return self._pandas_df.copy()

        def default_to_pandas(self, pandas_op, *args, **kwargs):
            return type(self)(pandas_op(self.to_pandas(), *args, **kwargs))
        
        def finalize(self):
            pass

        def free(self):
            pass

All done! Now you've got a fully functional query compiler, which is ready for extensions
and already can be used in Modin DataFrame:

.. code-block:: python

    import pandas
    pandas_df = pandas.DataFrame({"col1": [1, 2, 2, 1], "col2": [10, 2, 3, 40]})
    # Building our query compiler from pandas object
    qc = DefaultToPandasQueryCompiler.from_pandas(pandas_df)

    import modin.pandas as pd
    # Building Modin DataFrame from newly created query compiler
    modin_df = pd.DataFrame(query_compiler=qc)

    # Got fully functional Modin DataFrame
    >>> print(modin_df.groupby("col1").sum().reset_index())
       col1  col2
    0     1    50
    1     2     5

To be able to select this query compiler as default via ``modin.config`` you also need
to define the combination of your query compiler and pandas engine as an execution
by adding the corresponding factory. To find more information about factories,
visit :doc:`dispatching </flow/modin/core/execution/dispatching>` page.

Query Compiler API
''''''''''''''''''

.. autoclass:: modin.core.storage_formats.base.query_compiler.BaseQueryCompiler
    :members:
