:orphan:

Pandas storage format
"""""""""""""""""""""

.. toctree::
    :hidden:

    query_compiler
    parsers

High-Level Module Overview
''''''''''''''''''''''''''
This module houses submodules which are responsible for communication between
the query compiler level and execution implementation level for pandas storage format:

- :doc:`Query compiler <query_compiler>` is responsible for compiling efficient queries for :doc:`PandasDataframe </flow/modin/core/dataframe/pandas/dataframe>`.
- :doc:`Parsers <parsers>` are responsible for parsing data on workers during IO operations.
