Query compiler
==============

Modin has several execution backends. Calling any DataFrame API function will end up in
some backend-specific method. Query compiler is a bridge between Modin DataFrame and
an actual execution engine.

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

High-level module overview
''''''''''''''''''''''''''
This module houses submodules of all of the stable query compilers:

- :doc:`Base module<base/query_compiler>` contains an abstract query compiler class which defines common API.
- :doc:`Pandas module<pandas/index>` contains query compiler and text parsers for pandas backend.
- :doc:`Cudf module<cudf/index>` contains query compiler and text parsers for Cuda backend.
- :doc:`Pyarrow module<pyarrow/index>` contains query compiler and text parsers for Pyarrow backend.

You can find more in the :doc:`experimental section </flow/modin/experimental/backends/>`.
