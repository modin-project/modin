PyArrow query compiler
""""""""""""""""""""""
``PyarrowQueryCompiler`` is responsible for compiling efficient DataFrame algebra queries for the
:doc:`PyarrowOnRayFrame </flow/modin/experimental/engines/pyarrow_on_ray>`, the such frames that
are backed by ``pyarrow.Table`` objects.

Each ``PyarrowQueryCompiler`` contains an instance of ``PyarrowOnRayFrame`` which it queries to get the result.
