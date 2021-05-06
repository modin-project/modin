:orphan:

Base query compiler
"""""""""""""""""""

Brief description
'''''''''''''''''
``BaseQueryCompiler`` is an abstract class of query compiler, it sets a common interface,
that every other query compilers in Modin have to follow. Base class contains basic
implementations for most of the interface methods, all them are
:ref:`defaulting to pandas <defaulting-to-pandas-mechanism>`.

Subclassing ``BaseQueryCompiler``
'''''''''''''''''''''''''''''''''
If you want to add new type of query compiler to Modin you need to inherit new
class from `BaseQueryCompiler` and implement the rest of the abstract methods:

- ``from_pandas`` build query compiler from pandas DataFrame.
- ``from_arrow`` build query compiler from Arrow Table.
- ``to_pandas`` get query compilers representation via pandas DataFrame.
- ``default_to_pandas`` do :ref:`fallback to pandas <defaulting-to-pandas-mechanism>` for the passed function. 
- ``finalize`` finalize object constructing.
- ``free`` trigger memory cleaning.

(Please refer to the code documentation to see full contracts for these functions).

This is a minimum set of operations to ensure the proper work of your new query compiler,
the rest of the API will be defaulted pandas. To add backend-specific implementation for
some of the query compiler operation, just override the corresponding method in your
query compiler class.

Example
'''''''
As an exercise let's define a new query compiler in `Modin`, just to see how easy it is.
Usually, query compiler routes formed queries to the underlying :doc:`frame </flow/modin/engines/base/frame/data>` class,
which represents an actual execution engine and responsible for executing queries. In the glory
of simplicity and independence of this example, our execution engine will be the `pandas` itself.

So, we need to inherit a new class from ``BaseQueryCompiler`` and implement all of the abstract methods.
In case of `pandas` as an execution engine it's a trivial task:

.. code-block:: python

    from modin.backends import BaseQueryCompiler

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
to define the combination of your query compiler and pandas execution engine as a backend
by adding the corresponding factory. To find more information about factories,
visit :doc:`corresponding section </flow/modin/data_management/factories>` of the flow documentation.
