PyarrowQueryCompiler
""""""""""""""""""""
:py:class:`~modin.core.storage_formats.pyarrow.query_compiler.PyarrowQueryCompiler` is responsible for compiling efficient
Dataframe algebra queries for the :doc:`PyarrowOnRayDataframe </flow/modin/experimental/core/execution/ray/implementations/pyarrow_on_ray>`,
the frames which are backed by ``pyarrow.Table`` objects.

Each :py:class:`~modin.core.storage_formats.pyarrow.query_compiler.PyarrowQueryCompiler` contains an instance of
:py:class:`~modin.experimental.core.execution.ray.implementations.pyarrow_on_ray.dataframe.dataframe.PyarrowOnRayDataframe` which it queries to get the result.

Public API
''''''''''
:py:class:`~modin.core.storage_formats.pyarrow.query_compiler.PyarrowQueryCompiler` implements common query compilers API
defined by the :py:class:`~modin.core.storage_formats.base.query_compiler.BaseQueryCompiler`. Most functionalities
are inherited from :py:class:`~modin.core.storage_formats.pandas.query_compiler.PandasQueryCompiler`, in the following
section only overridden methods are presented.

.. autoclass:: modin.core.storage_formats.pyarrow.query_compiler.PyarrowQueryCompiler
  :members:
  :show-inheritance:
