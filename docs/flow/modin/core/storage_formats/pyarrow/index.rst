PyArrow storage format
""""""""""""""""""""""

.. toctree::
    :hidden:

    query_compiler
    parsers

In general, PyArrow storage formats follow the flow of the pandas ones: query compiler contains an instance of Modin Frame,
which is internally split into partitions. The main difference is that partitions contain PyArrow tables,
instead of DataFrames like with :doc:`pandas storage format </flow/modin/core/storage_formats/pandas/index>`. To learn more about this approach please
visit :doc:`PyArrow execution engine </flow/modin/experimental/core/execution/ray/implementations/pyarrow_on_ray>` section.


High-Level Module Overview
''''''''''''''''''''''''''
This module houses submodules which are responsible for communication between
the query compiler level and execution implementation level for PyArrow storage format:

- :doc:`Query compiler <query_compiler>` is responsible for compiling efficient queries for :doc:`PyarrowOnRayDataframe </flow/modin/experimental/core/execution/ray/implementations/pyarrow_on_ray>`.
- :doc:`Parsers <parsers>` are responsible for parsing data on workers during IO operations.

.. note::
    Currently the only one available PyArrow storage format factory is ``PyarrowOnRay`` which works
    in :doc:`experimental mode </flow/modin/experimental/experimental>` only.
