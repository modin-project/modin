:orphan:

OmniSciDB Execution Engine (experimental)
=========================================

OmniSciDB is an open-source SQL-based relational database designed for the
massive parallelism of modern CPU and GPU hardware. Its execution engine
is built on LLVM JIT compiler.

OmniSciDB can be embedded into an application as a dynamic library that
provides both C++ and Python APIs. A specialized in-memory storage layer
provides an efficient way to import data in Arrow table format.

Relational engine limitations
-----------------------------

Using a relational database engine implies a set of restrictions on
operations we can execute on a dataframe.

1. We cannot handle frames that use data types not supported by OmniSciDB.
Currently, we allow only integer, float, string, and categorical data types.

2. Column data should be homogeneous.

3. Can only support operations that map to relational algebra. This means
most operations are supported over a single axis (axis=0) only. Non-relational
operations like transposition and pivot are not supported.

When the unsupported data type is detected or unsupported operations is requested
we use the original pandas framework.

Lazy execution
--------------

OmniSciDB has a powerful query optimizer and an execution engine that
combines multiple operations into a single execution module. E.g. join,
filter and aggregation can be executed in a single data scan.

To utilize this feature and reduce data transformation and transfer
overheads, we need to implement lazy operations on a dataframe. The
dataframe with lazy computation is implemented in
:py:class:`~modin.experimental.core.execution.native.implementations.omnisci_on_native.dataframe.dataframe.OmnisciOnNativeDataframe`
class.

Lazy operations on a frame build a tree which is later translated into
a query executed by OmniSci. We use two types of trees. The first one
describes operations on frames that map to relational operations like
projection, union, etc. Nodes in this tree are derived from
:py:class:`~modin.experimental.core.execution.native.implementations.omnisci_on_native.df_algebra.DFAlgNode`
class. Some of the nodes (e.g.
:py:class:`~modin.experimental.core.execution.native.implementations.omnisci_on_native.df_algebra.TransformNode` mapped to a projection)
need a description of how individual columns are computed. The second
type of tree is used to describe operations on columns, including
arithmetic operations, type casts, datetime operations, etc. Nodes
of this tree are derived from
:py:class:`~modin.experimental.core.execution.native.implementations.omnisci_on_native.expr.BaseExpr` class.

Partitions
----------

Partitioning is used to achieve high parallelism. In the case of OmniSciDB
based execution parallelism is provided by OmniSciDB execution engine
and we don't need to manage multiple partitions.
:py:class:`~modin.experimental.core.execution.native.implementations.omnisci_on_native.dataframe.dataframe.OmnisciOnNativeDataframe`
always has a single partition.

A partition holds data in either ``pandas.DataFrame`` or ``pyarrow.Table``
format. ``pandas.DataFrame`` is preferred only when we detect unsupported
data type and therefore have to use ``pandas`` framework for processing.
In other cases ``pyarrow.Table`` format is preferred. Arrow tables can be
zero-copy imported into OmniSciDB. A query execution result is also
returned as an Arrow table.

Query execution
---------------

Frames are materialized (executed) when their data is accessed. E.g. it
happens when we try to access the frame's index or shape. We have two ways
to execute required operations.

Arrow execution
"""""""""""""""

For simple operations which don't include actual computations, we can use
Arrow API. We can use it to rename columns, drop columns and concatenate
frames.

OmniSciDB execution
"""""""""""""""""""

To execute query in OmniSciDB engine we need to import data first. We should
find all leaves of an operation tree and import their Arrow tables. Partitions
with imported tables hold corresponding table names used to refer to them in
queries.

OmniSciDB is SQL-based. SQL parsing is done in a separate process using
the Apache Calcite framework. A parsed query is serialized into JSON format
and is transferred back to OmniSciDB. In Modin, we don't generate SQL queries
for OmniSciDB but use this JSON format instead. Such queries can be directly
executed by OmniSciDB and also they can be transferred to Calcite server for
optimizations.

Operations used by Calcite in its intermediate representation are implemented
in classes derived from
:py:class:`~modin.experimental.core.execution.native.implementations.omnisci_on_native.calcite_algebra.CalciteBaseNode`.
:py:class:`~modin.experimental.core.execution.native.implementations.omnisci_on_native.calcite_builder.CalciteBuilder` is used to
translate :py:class:`~modin.experimental.core.execution.native.implementations.omnisci_on_native.df_algebra.DFAlgNode`-based
trees into :py:class:`~modin.experimental.core.execution.native.implementations.omnisci_on_native.calcite_algebra.CalciteBaseNode`-based sequences.
It also translates :py:class:`~modin.experimental.core.execution.native.implementations.omnisci_on_native.expr.BaseExpr`-based
trees by replacing :py:class:`~modin.experimental.core.execution.native.implementations.omnisci_on_native.expr.InputRefExpr`
nodes with either :py:class:`~modin.experimental.core.execution.native.implementations.omnisci_on_native.calcite_algebra.CalciteInputRefExpr`
or :py:class:`~modin.experimental.core.execution.native.implementations.omnisci_on_native.calcite_algebra.CalciteInputIdxExpr`
depending on context.

:py:class:`~modin.experimental.core.execution.native.implementations.omnisci_on_native.calcite_serializer.CalciteSerializer`
is used to serialize the resulting sequence into
JSON format. This JSON becomes a query by simply adding 'execute relalg'
or 'execute calcite' prefix (the latter is used if we want to use Calcite
for additional query optimization).

An execution result is a new Arrow table which is used to form a new
partition. This partition is assigned to the executed frame. The frame's
operation tree is replaced with
:py:class:`~modin.experimental.core.execution.native.implementations.omnisci_on_native.df_algebra.FrameNode` operation.

Column name mangling
''''''''''''''''''''

In ``pandas.DataFrame`` columns might have names not allowed in SQL (e. g.
an empty string). To handle this we simply add '`F_`' prefix to
column names. Index labels are more tricky because they might be non-unique.
Indexes are represented as regular columns, and we have to perform a special
mangling to get valid and unique column names. Demangling is done when we
transform our frame (i.e. its Arrow table) into ``pandas.DataFrame`` format.

Rowid column and sub-queries
''''''''''''''''''''''''''''

A special case of an index is the default index - 0-based numeric sequence.
In our representation, such an index is represented by the absence of index columns.
If we need to access the index value we can use the virtual ``rowid`` column provided
by OmniSciDB. Unfortunately, this special column is available for physical
tables only. That means we cannot access it for a node that is not a tree leaf.
That makes us execute trees with such nodes in several steps. First, we
materialize all frames that require ``rowid`` column and only after that we can
materialize the root of the tree.
