Modin vs. Dask Dataframe
========================

Dask's Dataframe is effectively a meta-frame, partitioning and scheduling many smaller
``pandas.DataFrame`` objects. The Dask DataFrame does not implement the entire pandas
API, and it isn't trying to. See this explained in the `Dask DataFrame documentation`_.

**The TL;DR is that Modin's API is identical to pandas, whereas Dask's is not. Note: The
projects are fundamentally different in their aims, so a fair comparison is
challenging.**

API
---
The API of Modin and Dask are different in several ways, explained here.

Dask DataFrame
""""""""""""""

Dask is currently missing multiple APIs from pandas that Modin has implemented. Of note:
Dask does not implement ``iloc``, ``MultiIndex``, ``apply(axis=0)``, ``quantile``,
``median``, and more. Some of these APIs cannot be implemented efficiently or at all
given the architecture design tradeoffs made in Dask's implementation, and others simply
require engineering effort. ``iloc``, for example, can be implemented, but it would be
inefficient, and ``apply(axis=0)`` cannot be implemented at all in Dask's architecture.

Dask DataFrames API is also different from the pandas API in that it is lazy and needs
``.compute()`` calls to materialize the DataFrame. This makes the API less convenient
but allows Dask to do certain query optimizations/rearrangement, which can give speedups
in certain situations. Several additional APIs exist in the Dask DataFrame API that
expose internal state about how the data is chunked and other data layout details, and
ways to manipulate that state.

Semantically, Dask sorts the ``index``, which does not allow for user-specified order.
In Dask's case, this was done for optimization purposes, to speed up other computations
which involve the row index.

Modin
"""""

Modin is targeted toward parallelizing the entire pandas API, without exception.
As the pandas API continues to evolve, so will Modin's pandas API. Modin is intended to
be used as a drop-in replacement for pandas, such that even if the API is not yet
parallelized, it still works by falling back to running pandas. One of the key features
of being a drop-in replacement is that not only will it work for existing code, if a
user wishing to go back to running pandas directly, they may at no cost. There's no
lock-in: Modin notebooks can be converted to and from pandas as the user prefers.

In the long-term, Modin is planned to become a data science framework that supports all
popular APIs (SQL, pandas, etc.) with the same underlying execution.

Architecture
------------

The differences in Modin and Dask's architectures are explained in this section.

Dask DataFrame
""""""""""""""

Dask DataFrame uses row-based partitioning, similar to Spark. This can be seen in their
`documentation`_. They also have a custom index object for indexing into the object,
which is not pandas compatible. Dask DataFrame seems to treat operations on the
DataFrame as MapReduce operations, which is a good paradigm for the subset of the pandas
API they have chosen to implement, but makes certain operations impossible. Dask
Dataframe is also lazy and places a lot of partitioning responsibility on the user.

Modin
"""""

Modin's partition is much more flexible, so the system can scale in both directions and
have finer grained partitioning. This is explained at a high level in `Modin's
documentation`_. Because we have this finer grained control over the partitioning, we
can support a number of operations that are very challenging in MapReduce systems (e.g.
transpose, median, quantile). This flexibility in partitioning also gives Modin
tremendous power to implement efficient straggler mitigation and improvements in
utilization over the entire cluster.

Modin is also architected to run on a variety of systems. The goal here is that users
can take the same notebook to different clusters or different environments and it will
still just work, run on what you have! Modin does support running on Dask's compute
engine in addition to Ray. The architecture of Modin is extremely modular, we are able
to add different execution engines or compile to different memory formats because of
this modularity. Modin can run on a Dask cluster in the same way that Dask Dataframe
can, but they will still be different in all of the ways described above.

Modin's implementation is grounded in theory, which is what enables us to implement the
entire pandas API.

.. _Dask DataFrame documentation: http://docs.dask.org/en/latest/dataframe.html#common-uses-and-anti-uses
.. _documentation: http://docs.dask.org/en/latest/dataframe.html#design.
.. _Modin's documentation: https://modin.readthedocs.io/en/latest/developer/architecture.html
