System Architecture
===================

In this section, we will lay out the overall system architecture for
Modin, as well as go into detail about the component design, implementation and
other important details. This document also contains important reference
information for those interested in contributing new functionality, bugfixes
and enhancements.

High-Level Architectural View
-----------------------------
The diagram below outlines the general layered view to the components of Modin
with a short description of each major section of the documentation following.


.. image:: /img/modin_architecture.png
   :align: center

Modin is logically separated into different layers that represent the hierarchy of a
typical Database Management System. Abstracting out each component allows us to
individually optimize and swap out components without affecting the rest of the system.
We can implement, for example, new compute kernels that are optimized for a certain type
of data and can simply plug it in to the existing infrastructure by implementing a small
interface. It can still be distributed by our choice of compute engine with the
logic internally.

System View
---------------------------
If we look to the overall class structure of the Modin system from very top, it will
look to something like this:

.. image:: /img/10000_meter.png
   :align: center

The user - Data Scientist interacts with the Modin system by sending interactive or
batch commands through API and Modin executes them using various backend execution
engines: Ray, Dask and MPI are currently supported.

Subsystem/Container View
------------------------
If we click down to the next level of details we will see that inside Modin the layered
architecture is implemented using several interacting components:

.. image:: /img/component_view.png
   :align: center

For the simplicity the other backend systems - Dask and MPI are omitted and only Ray backend is shown.

* Dataframe subsystem is the backbone of the dataframe holding and query compilation. It is responsible for
  dispatching the ingress/egress to the appropriate module, getting the Pandas API and calling the query
  compiler to convert calls to the internal intermediate Dataframe Algebra.
* Data Ingress/Egress Module is working in conjunction with Dataframe and Partitions subsystem to read data
  split into partitions and send data into the appropriate node for storing.
* Query Planner is subsystem that translates the Pandas API to intermediate Dataframe Algebra representation
  DAG and performs an initial set of optimizations.
* Query Executor is responsible for getting the Dataframe Algebra DAG, performing further optimizations based
  on a selected backend execution subsystem and mapping or compiling the Dataframe Algebra DAG to and actual
  execution sequence.
* Backends module is responsible for mapping the abstract operation to an actual executor call, e.g. Pandas,
  PyArrow, custom backend.
* Orchestration subsystem is responsible for spawning and controlling the actual execution environment for the
  selected backend. It spawns the actual nodes, fires up the execution environment, e.g. Ray, monitors the state
  of executors and provides telemetry

Component View
--------------
Coming soon...

Module/Class View
-----------------
Coming soon...

DataFrame Partitioning
----------------------

The Modin DataFrame architecture follows in the footsteps of modern architectures for
database and high performance matrix systems. We chose a partitioning schema that
partitions along both columns and rows because it gives Modin flexibility and
scalability in both the number of columns and the number of rows supported. The
following figure illustrates this concept.

.. image:: /img/block_partitions_diagram.png
   :align: center

Currently, each partition's memory format is a `pandas DataFrame`_. In the future, we will
support additional in-memory formats for the backend, namely `Arrow tables`_.

Index
"""""

We currently use the ``pandas.Index`` object for both indexing columns and rows. In the
future, we will implement a distributed, pandas-compatible Index object in order remove
this scaling limitation from the system. It does not start to become a problem until you
are operating on more than 10's of billions of columns or rows, so most workloads will
not be affected by this scalability limit. **Important note**: If you are using the
default index (``pandas.RangeIndex``) there is a fixed memory overhead (~200 bytes) and
there will be no scalability issues with the index.


API
"""

The API is the outer-most layer that faces users. The majority of our current effort is
spent implementing the components of the pandas API. We have implemented a toy example
for a sqlite API as a proof of concept, but this isn't ready for usage/testing. There
are also plans to expose the Modin DataFrame API as a reduced API set that encompasses
the entire pandas/dataframe API.

Query Compiler
""""""""""""""

The Query Compiler receives queries from the pandas API layer. The API layer's
responsibility is to ensure clean input to the Query Compiler. The Query Compiler must
have knowledge of the compute kernels/in-memory format of the data in order to
efficiently compile the queries.

The Query Compiler is responsible for sending the compiled query to the Modin DataFrame.
In this design, the Query Compiler does not have information about where or when the
query will be executed, and gives the control of the partition layout to the Modin
DataFrame.

In the interest of reducing the pandas API, the Query Compiler layer closely follows the
pandas API, but cuts out a large majority of the repetition.

Modin DataFrame
"""""""""""""""

At this layer, operations can be performed lazily. Currently, Modin executes most
operations eagerly in an attempt to behave as pandas does. Some operations, e.g.
``transpose`` are expensive and create full copies of the data in-memory. In these
cases, we can wait until another operation triggers computation. In the future, we plan
to add additional query planning and laziness to Modin to ensure that queries are
performed efficiently.

The structure of the Modin DataFrame is extensible, such that any operation that could
be better optimized for a given backend can be overridden and optimized in that way.

This layer has a significantly reduced API from the QueryCompiler and the user-facing
API. Each of these APIs represents a single way of performing a given operation or
behavior. Some of these are expanded for convenience/understanding. The API abstractions
are as follows:

Modin DataFrame API
'''''''''''''''''''

* ``mask``: Indexing/masking/selecting on the data (by label or by integer index).
* ``copy``: Create a copy of the data.
* ``mapreduce``: Reduce the dimension of the data.
* ``foldreduce``: Reduce the dimension of the data, but entire column/row information is needed.
* ``map``: Perform a map.
* ``fold``: Perform a fold.
* ``apply_<type>``: Apply a function that may or may not change the shape of the data.

   * ``full_axis``: Apply a function requires knowledge of the entire axis.
   * ``full_axis_select_indices``: Apply a function performed on a subset of the data that requires knowledge of the entire axis.
   * ``select_indices``: Apply a function to a subset of the data. This is mainly used for indexing.

* ``binary_op``: Perform a function between two dataframes.
* ``concat``: Append one or more dataframes to either axis of this dataframe.
* ``transpose``: Swap the axes (columns become rows, rows become columns).
* ``groupby``:

   * ``groupby_reduce``: Perform a reduction on each group.
   * ``groupby_apply``: Apply a function to each group.

* take functions
   * ``head``: Take the first ``n`` rows.
   * ``tail``: Take the last ``n`` rows.
   * ``front``: Take the first ``n`` columns.
   * ``back``: Take the last ``n`` columns.

* import/export functions
   * ``from_pandas``: Convert a pandas dataframe to a Modin dataframe.
   * ``to_pandas``: Convert a Modin dataframe to a pandas dataframe.
   * ``to_numpy``: Convert a Modin dataframe to a numpy array.

More documentation can be found internally in the code_. This API is not complete, but
represents an overwhelming majority of operations and behaviors.

This API can be implemented by other distributed/parallel DataFrame libraries and
plugged in to Modin as well. Create an issue_ or discuss on our Discourse_ for more
information!

The Modin DataFrame is responsible for the data layout and shuffling, partitioning,
and serializing the tasks that get sent to each partition. Other implementations of the
Modin DataFrame interface will have to handle these as well.

Execution Engine/Framework
""""""""""""""""""""""""""

This layer is what Modin uses to perform computation on a partition of the data. The
Modin DataFrame is designed to work with `task parallel`_ frameworks, but with some
effort, a data parallel framework is possible.

Internal abstractions
"""""""""""""""""""""

These abstractions are not included in the above architecture, but are important to the
internals of Modin.

Partition Manager
'''''''''''''''''

The Partition Manager can change the size and shape of the partitions based on the type
of operation. For example, certain operations are complex and require access to an
entire column or row. The Partition Manager can convert the block partitions to row
partitions or column partitions. This gives Modin the flexibility to perform operations
that are difficult in row-only or column-only partitioning schemas.

Another important component of the Partition Manager is the serialization and shipment
of compiled queries to the Partitions. It maintains metadata for the length and width of
each partition, so when operations only need to operate on or extract a subset of the
data, it can ship those queries directly to the correct partition. This is particularly
important for some operations in pandas which can accept different arguments and
operations for different columns, e.g. ``fillna`` with a dictionary.

This abstraction separates the actual data movement and function application from the
DataFrame layer to keep the DataFrame API small and separately optimize the data
movement and metadata management.

Partition
'''''''''

Partitions are responsible for managing a subset of the DataFrame. As is mentioned
above, the DataFrame is partitioned both row and column-wise. This gives Modin
scalability in both directions and flexibility in data layout. There are a number of
optimizations in Modin that are implemented in the partitions. Partitions are specific
to the execution framework and in-memory format of the data. This allows Modin to
exploit potential optimizations across both of these. These optimizations are explained
further on the pages specific to the execution framework.

Supported Execution Frameworks and Memory Formats
"""""""""""""""""""""""""""""""""""""""""""""""""

This is the list of execution frameworks and memory formats supported in Modin. If you
would like to contribute a new execution framework or memory format, please see the
documentation page on :doc:`../contributing`.

- `Pandas on Ray`_
    - Uses the Ray_ execution framework.
    - The compute kernel/in-memory format is a pandas DataFrame.
- `Pandas on Dask`_
    - Uses the `Dask Futures`_ execution framework.
    - The compute kernel/in-memory format is a pandas DataFrame.
- `Pyarrow on Ray`_ (experimental)
    - Uses the Ray_ execution framework.
    - The compute kernel/in-memory format is a pyarrow Table.

.. _pandas Dataframe: https://pandas.pydata.org/pandas-docs/version/0.23.4/generated/pandas.DataFrame.html
.. _Arrow tables: https://arrow.apache.org/docs/python/generated/pyarrow.Table.html
.. _Ray: https://github.com/ray-project/ray
.. _code: https://github.com/modin-project/modin/blob/master/modin/engines/base/frame/data.py
.. _Contributing: contributing.html
.. _Pandas on Ray: UsingPandasonRay/optimizations.html
.. _Pandas on Dask: UsingPandasonDask/optimizations.html
.. _Dask Futures: https://docs.dask.org/en/latest/futures.html
.. _issue: https://github.com/modin-project/modin/issues
.. _Discourse: https://discuss.modin.org
.. _task parallel: https://en.wikipedia.org/wiki/Task_parallelism
.. _Pyarrow on Ray: UsingPyarrowonRay/index.html
