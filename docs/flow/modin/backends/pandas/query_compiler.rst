Pandas Query Compiler
"""""""""""""""""""""
:py:class:`~modin.backends.pandas.query_compiler.PandasQueryCompiler` is responsible for compiling
a set of known predefined functions and pairing those with dataframe algebra operators in the
:doc:`PandasFrame </flow/modin/engines/base/frame/data>`, specifically for dataframes backed by
``pandas.DataFrame`` objects.

Each :py:class:`~modin.backends.pandas.query_compiler.PandasQueryCompiler` contains an instance of
:py:class:`~modin.engines.base.frame.data.PandasFrame` which it queries to get the result.

:py:class:`~modin.backends.pandas.query_compiler.PandasQueryCompiler` supports methods built by the :doc:`function module </flow/modin/data_management/functions>`.
If you want to add an implementation for a query compiler method, visit the function module documentation
to see whether the new operation fits one of the existing function templates and can be easily implemented
with them.

Public API
''''''''''
:py:class:`~modin.backends.pandas.query_compiler.PandasQueryCompiler` implements common query compilers API
defined by the :py:class:`~modin.backends.base.query_compiler.BaseQueryCompiler`. Some functionalities
are inherited from the base class, in the following section only overridden methods are presented.

.. autoclass:: modin.backends.pandas.query_compiler.PandasQueryCompiler
  :members:
