Query Compiler
==============

.. toctree::
    :hidden:

    base/query_compiler
    pandas/index
    cudf/index
    pyarrow/index

Modin supports several execution backends. Calling any DataFrame API function will end up in
some backend-specific method. The query compiler is a bridge between Modin Dataframe and
an actual execution engine.

.. image:: /img/simplified_query_flow.svg
    :align: right
    :width: 300px

Query compilers of all backends implement a common API, which is used by the Modin Dataframe
to support dataframe queries. The role of the query compiler is to translate its API into
a pairing of known user-defined functions and dataframe algebra operators. Each query compiler instance contains a
:doc:`frame </flow/modin/engines/base/frame/index>` of the selected execution engine and queries
it with the compiled queries to get the result. The query compiler object is immutable,
so the result of every method is a new query compiler.

The query compilers API is defined by the :py:class:`~modin.backends.base.query_compiler.BaseQueryCompiler` class
and may resemble the pandas API, however, they're not equal. The query compilers API
is significantly reduced in comparison with pandas, since many corner cases or even the
whole methods can be handled at the API layer with the existing API.

The query compiler is the level where Modin stops distinguishing DataFrame and Series (or column) objects.
A Series is represented by a `1xN` query compiler, where the Series name is the column label.
If Series is unnamed, then the label is ``"__reduced__"``. The Dataframe API layer
interprets a one-column query compiler as Series or DataFrame depending on the operation context.

.. note::
    Although we're declaring that there is no difference between DataFrame and Series at the query compiler,
    you still may find methods like ``method_ser`` and ``method_df`` which are implemented differently because they're
    emulating either Series or DataFrame logic, or you may find parameters, which indicates whether this one-column
    query compiler is representing Series or not. All of these are hacks, and we're working on getting rid of them.

High-level module overview
''''''''''''''''''''''''''
This module houses submodules of all of the stable query compilers:

- :doc:`Base module<base/query_compiler>` contains an abstract query compiler class which defines common API.
- :doc:`Pandas module<pandas/index>` contains query compiler and text parsers for pandas backend.
- :doc:`cuDF module<cudf/index>` contains query compiler and text parsers for cuDF backend.
- :doc:`Pyarrow module<pyarrow/index>` contains query compiler and text parsers for Pyarrow backend.

You can find more in the :doc:`experimental section </flow/modin/experimental/backends/>`.
