PyArrow Query Compiler
""""""""""""""""""""""
:py:class:`~modin.backends.pyarrow.query_compiler.PyarrowQueryCompiler` is responsible for compiling efficient
DataFrame algebra queries for the :doc:`PyarrowOnRayFrame </flow/modin/experimental/engines/pyarrow_on_ray>`, 
the such frames that are backed by ``pyarrow.Table`` objects.

Each :py:class:`~modin.backends.pyarrow.query_compiler.PyarrowQueryCompiler` contains an instance of
:py:class:`~modin.experimental.engines.pyarrow_on_ray.frame.data.PyarrowOnRayFrame` which it queries to get the result.

Public API
''''''''''
:py:class:`~modin.backends.pyarrow.query_compiler.PyarrowQueryCompiler` implements common query compilers API
defined by the :py:class:`~modin.backends.base.query_compiler.BaseQueryCompiler`. Most functionalities
are inherited from :py:class:`~modin.backends.pandas.query_compiler.PandasQueryCompiler`, in the following
section only overridden methods are presented.

.. autoclass:: modin.backends.pyarrow.query_compiler.PyarrowQueryCompiler
  :members:
