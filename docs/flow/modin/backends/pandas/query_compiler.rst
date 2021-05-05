Pandas query compiler
"""""""""""""""""""""
`PandasQueryCompiler` is responsible for compiling efficient DataFrame algebra queries for the
:doc:`BasePandasFrame </flow/modin/engines/base/frame/data>`, the such frame that contains
pandas DataFrames in its partitions as a payload.

Each `PandasQueryCompiler` contains an instance of `BasePandasFrame` which it queries to get the result.

`PandasQueryCompiler` supports methods built by :doc:`function module </flow/modin/data_management/functions>`,
so if you want to add implementation for some of the query compilers API method you may visit this module
to inspect whether the new operation suits one of the presented templates and can be easily implemented
with them.
