Query compiler
==============

Modin has several execution backends. Calling any DataFrame API function will end up in
some backend-specific method. Query compiler is a bridge between Modin DataFrame and
an actual execution engine.

.. image:: /img/simplified_query_flow.svg
    :align: right
    :width: 300px

Query compilers of all backends provide a common API, which is used by Modin DataFrame
to emulate pandas API. The role of the query compiler is to translate its API into
a set of DataFrame algebra operations. Each query compiler instance contains
:doc:`frame </flow/modin/engines/base/frame/index>` of the selected execution engine and query
it with the compiled queries to get the result. The query compiler object is immutable,
so the result of every method is a new query compiler.

Query compilers API is defined by the :doc:`BaseQueryCompiler <base/query_compiler>` class
and partially mimic the pandas API, however, they're not equal. Query compilers API
is significantly reduced in comparison with pandas, since many corner cases or even the
whole methods can be handled at the DataFrame level with the existing API.

Query compiler is the level where Modin stops distinguishing Frame and Series objects.
Series is represented by one-column query compiler, where Series name is the column label,
if Series is unnamed, then the label would be ``"__reduced__"``. DataFrame API level
interprets one-column query compilers as Series or DataFrame depending on the operation context.

.. note::
    Although we're declaring that there is no difference between DataFrame and Series at the query compiler,
    you still may find methods like ``method_ser`` and ``method_df`` which perform differently because they
    emulating either Series or DataFrame logic, or you may find parameters, which indicates whether this one-column
    query compiler is representing Series or not. All of these are hacks, and we're working on getting rid of them.

High-level module overview
''''''''''''''''''''''''''
This module houses submodules of all of the stable query compilers:

- :doc:`Base module<base/query_compiler>` contains an abstract query compiler class which defines common API.
- :doc:`Pandas module<pandas/index>` contains query compiler and text parsers for pandas backend.
- :doc:`Cudf module<cudf/index>` contains query compiler and text parsers for Cuda backend.
- :doc:`Pyarrow module<pyarrow/index>` contains query compiler and text parsers for Pyarrow backend.

You can find more in the :doc:`experimental section </flow/modin/experimental/backends/>`.
