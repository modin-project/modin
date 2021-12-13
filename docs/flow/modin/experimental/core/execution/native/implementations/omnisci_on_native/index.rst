:orphan:

OmnisciOnNative execution
=========================

OmniSciDB is an open-source SQL-based relational database designed for the
massive parallelism of modern CPU and GPU hardware. Its execution engine
is built on LLVM JIT compiler.

OmniSciDB can be embedded into an application as a dynamic library that
provides both C++ and Python APIs. A specialized in-memory storage layer
provides an efficient way to import data in Arrow table format.

`OmnisciOnNative` execution uses OmniSciDB for both as a storage format and for
actual data transformation.

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
it falls back to the original pandas framework.

Partitions
----------

In Modin, partitioning is used to achieve high parallelism. In the case of 
OmniSciDB-based execution, parallelism is provided by OmniSciDB execution
engine itself and we don't need to manage multiple partitions.
:py:class:`~modin.experimental.core.execution.native.implementations.omnisci_on_native.dataframe.dataframe.OmnisciOnNativeDataframe`
always has a single partition.

A partition holds data in either ``pandas.DataFrame`` or ``pyarrow.Table``
format. ``pandas.DataFrame`` is preferred only when we detect unsupported
data type and therefore have to use ``pandas`` framework for processing.
In other cases ``pyarrow.Table`` format is preferred. Arrow tables can be
zero-copy imported into OmniSciDB. A query execution result is also
returned as an Arrow table.

Data Ingress
------------

When users import data in Modin DataFrame (from a file or from some python
object like array or dictionary) they invoke one of the ``modin.pandas.io`` 
functions (to read data from a file) or use :py:class:`~modin.pandas.dataframe.DataFrame` constructor
(to create a DataFrame from an iterable object). Both of the paths lead to the
:py:class:`~modin.core.execution.dispatching.factories.dispatcher.FactoryDispatcher`
that defines a factory that handles the import query. For `OmnisciOnNative`
execution, the factory is accordingly 
:py:class:`~modin.core.execution.dispatching.factories.factories.ExperimentalOmnisciOnNativeFactory`.
The factory dispatches the import query: if the data needs to be read from a file
- the query is routed to the 
:py:class:`~modin.experimental.core.execution.native.implementations.omnisci_on_native.io.OmnisciOnNativeIO`
class, that uses Arrow Framework to read the file into a PyArrow Table, the resulted
table is passed to the 
:py:class:`~modin.experimental.core.storage_formats.omnisci.query_compiler.DFAlgQueryCompiler`.
If the factory deals with importing a Python's iterable object, the query goes straight
into the 
:py:class:`~modin.experimental.core.storage_formats.omnisci.query_compiler.DFAlgQueryCompiler`.
The Query Compiler sanitizes an input object and passes it to one of the
:py:class:`~modin.experimental.core.execution.native.implementations.omnisci_on_native.dataframe.dataframe.OmnisciOnNativeDataframe` 
factory methods (``.from_*``). The Datarame's build method stores the passed object into a new Dataframe's partition
and returns the resulted Dataframe, which is then wrapped into a Query Compiler, which is
wrapped into a high-level Modin DataFrame, which is returned to the user.

.. figure:: /img/omnisci/omnisci_ingress.svg
   :align: center

Note that during this ingress flow, no data is actually imported to the OmniSciDB. The need for
importing to OmniSci is decided later at the execution stage by the Modin Core Dataframe layer.
If the query requires for the data to be placed in OmniSciDB, the import is triggered.
:py:class:`~modin.experimental.core.execution.native.implementations.omnisci_on_native.dataframe.dataframe.OmnisciOnNativeDataframe` 
passes partition to import to the
:py:class:`~modin.experimental.core.execution.native.implementations.omnisci_on_native.partitioning.partition_manager.OmnisciOnNativeDataframePartitionManager`
that extracts a partition's underlying object and sends a request to import it to the OmniSci
Server. The response for the request is a unique identifier for the just imported table
at OmniSciDB, this identifier is placed in the partition. After that, the partition has
a reference to the concrete table in OmniSciDB to query, and the data is considered to be
fully imported.

.. figure:: /img/omnisci/omnisci_import.svg
   :align: center

Data Transformation
-------------------

.. figure:: /img/omnisci/omnisci_query_flow.svg
   :align: center

When a user calls any :py:class:`~modin.pandas.dataframe.DataFrame` API, a query
starts forming at the `API` layer to be executed at the `Execution` layer. The `API`
layer is responsible for processing the query appropriately, for example, determining
whether the final result should be a ``DataFrame`` or ``Series`` object, and
sanitizing the input to the
:py:class:`~modin.experimental.core.storage_formats.omnisci.query_compiler.DFAlgQueryCompiler`,
e.g. validating a parameter from the query and defining specific intermediate values
to provide more context to the query compiler.

The :py:class:`~modin.experimental.core.storage_formats.omnisci.query_compiler.DFAlgQueryCompiler`
is responsible for reducing the recieved query to the pre-defined Dataframe algebra operators
and pass their execution to the
:py:class:`~modin.experimental.core.execution.native.implementations.omnisci_on_native.dataframe.dataframe.OmnisciOnNativeDataframe`.

When :py:class:`~modin.experimental.core.execution.native.implementations.omnisci_on_native.dataframe.dataframe.OmnisciOnNativeDataframe`
recieves a query it determines whether the operation requires data materialization
or can be performed lazily. Depending on that the operation is either appended to a
lazy computation tree or executed.

Lazy execution
""""""""""""""

OmniSciDB has a powerful query optimizer and an execution engine that
combines multiple operations into a single execution module. E.g. join,
filter and aggregation can be executed in a single data scan.

To utilize this feature and reduce data transformation and transfer
overheads, all of the operations that don't require data materialization
are performed lazily.

Lazy operations on a frame build a tree which is later translated into
a query executed by OmniSci. Each of the tree nodes has its input node(s)
- a frame argument(s) of the operation. When a new node is appended to the
tree, it becomes its root. The leaves of the tree are always a special node
type, whose input is an actual materialized frame to execute operations 
from the tree on.

.. figure:: /img/omnisci/omnisci_lazy_tree_example.svg
   :align: center

There are two types of trees. The first one describes operations on frames that
map to relational operations like projection, union, etc. Nodes in this tree are
derived from
:py:class:`~modin.experimental.core.execution.native.implementations.omnisci_on_native.df_algebra.DFAlgNode`
class. Leaf nodes are instances of the
:py:class:`~modin.experimental.core.execution.native.implementations.omnisci_on_native.df_algebra.FrameNode`
class. The second type of tree is used to describe operations on columns, including
arithmetic operations, type casts, datetime operations, etc. Nodes of this tree are derived from
:py:class:`~modin.experimental.core.execution.native.implementations.omnisci_on_native.expr.BaseExpr`
class. Leaf nodes are instances of the
:py:class:`~modin.experimental.core.execution.native.implementations.omnisci_on_native.expr.InputRefExpr`
class.

Visit the corresponding sections to go through all of the types of nodes:

* :doc:`Frame nodes <df_algebra>`
* :doc:`Expression nodes <expr>`

Execution of a computation tree
"""""""""""""""""""""""""""""""

Frames are materialized (executed) when their data is accessed. E.g. it
happens when we try to access the frame's index or shape. There are two ways
to execute required operations: through Arrow or through OmniSciDB.

Arrow execution
'''''''''''''''

For simple operations which don't include actual computations, execution can use
Arrow API. We can use it to rename columns, drop columns and concatenate
frames. Arrow execution is preferable since it doesn't require actual data import/export
to the OmniSciDB.

OmniSciDB execution
'''''''''''''''''''

To execute query in OmniSciDB engine we need to import data first. We should
find all leaves of an operation tree and import their Arrow tables. Partitions
with imported tables hold corresponding table names used to refer to them in
queries.

OmniSciDB is SQL-based. SQL query parsing is done in a separate process using
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

.. figure:: /img/omnisci/omnisci_calcite_serialization_flow.svg
   :align: center

The building of Calcite query (starting from the conversion to the Calcite Algebra and up to
the forming JSON query) is orchestrated by 
:py:class:`~modin.experimental.core.execution.native.implementations.omnisci_on_native.partitioning.partition_manager.OmnisciOnNativeDataframePartitionManager`.

An execution result is a new Arrow table which is used to form a new
partition. This partition is assigned to the executed frame. The frame's
operation tree is replaced with
:py:class:`~modin.experimental.core.execution.native.implementations.omnisci_on_native.df_algebra.FrameNode` operation.

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

OmnisciOnNative Dataframe Implementation
----------------------------------------

Modin implements ``Dataframe``, ``PartitionManager`` and ``Partition`` classes
specific for ``OmnisciOnNative`` execution:

* :doc:`OmnisciOnNativeDataframe <dataframe>`
* :doc:`OmnisciOnNativeDataframePartition <partitioning/partition>`
* :doc:`OmnisciOnNativeDataframePartitionManager <partitioning/partition_manager>`

To support lazy execution Modin uses two types of trees. Operations on frames are described
by ``DFAlgNode`` based trees. Scalar computations are described by ``BaseExpr`` based tree.

* :doc:`Frame nodes <df_algebra>`
* :doc:`Expression nodes <expr>`

Interactions with OmniSci engine are done using ``OmnisciServer`` class. Queries use serialized
Calcite relational algebra format. Calcite algebra nodes are based on ``CalciteBaseNode`` class.
Translation is done by ``CalciteBuilder`` class. Serialization is performed by ``CalciteSerializer``
class.

* :doc:`CalciteBaseNode <calcite_algebra>`
* :doc:`CalciteBuilder <calcite_builder>`
* :doc:`CalciteSerializer <calcite_serializer>`
* :doc:`OmnisciServer <omnisci_worker>`

Column name mangling
""""""""""""""""""""

In ``pandas.DataFrame`` columns might have names not allowed in SQL (e. g.
an empty string). To handle this we simply add '`F_`' prefix to
column names. Index labels are more tricky because they might be non-unique.
Indexes are represented as regular columns, and we have to perform a special
mangling to get valid and unique column names. Demangling is done when we
transform our frame (i.e. its Arrow table) into ``pandas.DataFrame`` format.

.. toctree::
    :hidden:

    dataframe
    partitioning/partition
    partitioning/partition_manager
    df_algebra
    expr
    calcite_algebra
    calcite_builder
    calcite_serializer
    omnisci_worker
