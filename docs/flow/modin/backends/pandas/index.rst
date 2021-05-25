:orphan:

Pandas backend
""""""""""""""

.. toctree::
    :hidden:

    query_compiler
    parsers

High-Level Module Overview
''''''''''''''''''''''''''
This module houses submodules which are responsible for communication between
the query compiler level and execution engine level for pandas backend:

- :doc:`Query compiler <query_compiler>` is responsible for compiling efficient queries for :doc:`BasePandasFrame </flow/modin/engines/base/frame/data>`.
- :doc:`Parsers <parsers>` are responsible for parsing data on workers during IO operations.
