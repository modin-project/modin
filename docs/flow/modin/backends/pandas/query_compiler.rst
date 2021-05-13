Pandas query compiler
"""""""""""""""""""""
``PandasQueryCompiler`` is responsible for compiling efficient DataFrame algebra queries for the
:doc:`BasePandasFrame </flow/modin/engines/base/frame/data>`, specifically for dataframes backed by
``pandas.DataFrame`` objects.

Each ``PandasQueryCompiler`` contains an instance of ``BasePandasFrame`` which it queries to get the result.

``PandasQueryCompiler`` supports methods built by the :doc:`function module </flow/modin/data_management/functions>`.
If you want to add an implementation for a query compiler method, visit the function module documentation
to see whether the new operation fits one of the existing function templates and can be easily implemented
with them.
