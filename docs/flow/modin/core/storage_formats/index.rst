:orphan:

Storage Formats
===============
Storage format is one of the components that form Modin's execution, it describes the type(s)
of objects that are stored in the partitions of the selected Core Modin Dataframe implementation.

The base storage format in Modin is pandas. In that format, Modin Dataframe operates with
partitions that hold ``pandas.DataFrame`` objects. Pandas is the most natural storage format
since high-level DataFrame objects mirror its API.

The storage format + execution engine (Ray, Dask, etc.) form the execution backend. 
The Query Compiler (QC) converts high-level pandas API calls to queries that are understood 
by the execution backend.

.. _query_compiler_def:

Query Compiler
==============

.. toctree::
    :hidden:

    base/query_compiler
    pandas/index

Modin supports several execution backends (storage format + execution engine). Calling any
DataFrame API function will end up in some execution-specific method. The query compiler is
a bridge between pandas DataFrame API and the actual Core Modin Dataframe implementation for the
corresponding execution.

.. image:: /img/simplified_query_flow.svg
    :align: right
    :width: 300px

Each storage format has its own Query Compiler class that implements the most optimal
query routing for the selected format.

Query compilers of all storage formats implement a common API, which is used by the high-level Modin DataFrame
to support dataframe queries. The role of the query compiler is to translate its API into
a pairing of known user-defined functions and dataframe algebra operators. Each query compiler instance contains a
:doc:`Core Modin Dataframe </flow/modin/core/dataframe/base/index>` of the selected execution implementation and queries
it with the compiled queries to get the result. The query compiler object is immutable,
so the result of every method is a new query compiler.

The query compilers API is defined by the :py:class:`~modin.core.storage_formats.base.query_compiler.BaseQueryCompiler` class
and may resemble the pandas API, however, they're not equal. The query compilers API
is significantly reduced in comparison with pandas, since many corner cases or even the
whole methods can be handled at the API layer with the existing API.

The query compiler is the level where Modin stops distinguishing DataFrame and Series (or column) objects.
A Series is represented by a `1xN` query compiler, where the Series name is the column label.
If Series is unnamed, then the label is ``MODIN_UNNAMED_SERIES_LABEL``, which is equal to ``"__reduced__"``. The high-level DataFrame API layer
interprets a one-column query compiler as Series or DataFrame depending on the operation context.

.. note::
    Although we're declaring that there is no difference between DataFrame and Series at the query compiler,
    you still may find methods like ``method_ser`` and ``method_df`` which are implemented differently because they're
    emulating either Series or DataFrame logic, or you may find parameters, which indicates whether this one-column
    query compiler is representing Series or not. All of these are hacks, and we're working on getting rid of them.

High-level module overview
''''''''''''''''''''''''''

This module houses submodules of all of the stable storage formats:

- :doc:`Base module <base/query_compiler>` contains an abstract query compiler class which defines common API.
- :doc:`Pandas module <pandas/index>` contains query compiler and text parsers for pandas storage format.
