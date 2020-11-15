# What is the difference between Dask DataFrame and Modin?

**The TL;DR is that Modin's API is identical to pandas, whereas Dask's is not. Note: The projects are fundamentally different in their aims, so a fair comparison is challenging.**

## API

### Dask DataFrame

Dask DataFrame does not scale the entire pandas API, and it isn't trying to. See this explained in their documentation [here](http://docs.dask.org/en/latest/dataframe.html#common-uses-and-anti-uses)

Dask DataFrames API is also different from the pandas API in that it is lazy and needs .compute() to materialize the DataFrame. This makes the API less convenient but allows to do certain query optimizations/rearrangement, which can give speedups in certain situations. We are planning to incorporate similar capabilities into Modin but hope we can do so without having to change the API. We will outline plans for speeding up Modin in an upcoming blog post.

### Modin

Modin attempts to parallelize as much of the pandas API as is possible. We have worked through a significant portion of the DataFrame API. It is intended to be used as a drop-in replacement for pandas, such that even if the API is not yet parallelized, it is still defaulting to pandas.

## Architecture

### Dask DataFrame

Dask DataFrame has row-based partitioning, similar to Spark. This can be seen in their [documentation](http://docs.dask.org/en/latest/dataframe.html#design.) They also have a custom index object for indexing into the object, which is not pandas compatible. Dask DataFrame seems to treat operations on the DataFrame as MapReduce operations, which is a good paradigm for the subset of the pandas API they have chosen to implement.

### Modin

Modin is more of a column-store, which we inherited from modern database systems. We laterally partition the columns for scalability (many systems, such as Google BigTable already did this), so we can scale in both directions and have finer grained partitioning. This is explained at a high level in [Modin's documentation](https://modin.readthedocs.io/en/latest/architecture.html). Because we have this finer grained control over the partitioning, we can support a number of operations that are very challenging in MapReduce systems (e.g. transpose, median, quantile).

## Modin aims

In the long-term, Modin is planned to become a DataFrame library that supports the popular APIs (SQL, pandas, etc.) and runs on a variety of compute engines and backends. In fact, a group was able to contribute a dask.delayed backend to Modin already in <200 lines of code [PR](https://github.com/modin-project/modin/pull/281).


- Reference: [Query: What is the difference between Dask and Modin? #515](https://github.com/modin-project/modin/issues/515) 