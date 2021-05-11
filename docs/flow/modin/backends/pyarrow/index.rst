:orphan:

Pyarrow backend
"""""""""""""""
In general, pyarrow backend follows the flow of pandas backend: query compilers contain an instance of Modin Frame,
which is internally split into partitions. The main difference is that partitions contain pyarrow tables,
instead of DataFrames like in pandas backend. To learn more about this approach please
visit :doc:`pyarrow execution engine </flow/modin/experimental/engines/pyarrow_on_ray>` section.


High-Level Module Overview
''''''''''''''''''''''''''
This module houses submodules which are responsible for communication between
the query compiler level and execution engine level for pyarrow backend:

- :doc:`Query compiler <query_compiler>` is responsible for compiling efficient queries for :doc:`PyarrowOnRayFrame </flow/modin/experimental/engines/pyarrow_on_ray>`.
- :doc:`Parsers <parsers>` are responsible for parsing data on workers during IO operations.

.. note::
    Currently the only one available pyarrow backend is ``PyarrowOnRay`` which works
    in :doc:`experimental mode </flow/modin/experimental/experimental>` only.
