
Modin utilizes a range-partitioning approach for specific operations, significantly enhancing
parallelism and reducing memory consumption in certain scenarios.

You can enable range-partitioning by specifying ``cfg.RangePartitioning`` :doc:`configuration variable: </flow/modin/config>`:

.. code-block:: python
    import modin.pandas as pd
    import modin.config as cfg

    cfg.RangePartitioning.put(True) # past this point methods that support range-partitioning
                                    # will use engage it

    pd.DataFrame(...).groupby(...).mean() # use range-partitioning for groupby.mean()

    cfg.Range-partitioning.put(False)

    pd.DataFrame(...).groupby(...).mean() # use MapReduce implementation for groupby.mean()

Building range-partitioning assumes data reshuffling, which may result into order of rows different from
pandas in certain operations.

Range-partitioning is not a silver bullet, meaning that enabling it is not always beneficial. Below you find
a list of operations that have support for range-partitioning and practical advices on when one should
enable it.

Range-partitioning GroupBy
""""""""""""""""""""""""""

TODO: rewrite this section

The range-partitioning GroupBy implementation utilizes Modin's reshuffling mechanism that gives an
ability to build range partitioning over a Modin DataFrame.

In order to enable/disable the range-partitiong implementation you have to specify ``cfg.RangePartitioning``
:doc:`configuration variable: </flow/modin/config>`

.. code-block:: ipython

    In [4]: import modin.config as cfg; cfg.RangePartitioning.put(True)

    In [5]: # past this point, Modin will always use the range-partitiong groupby implementation

    In [6]: cfg.RangePartitioning.put(False)

    In [7]: # past this point, Modin won't use range-partitiong groupby implementation anymore

The range-partitiong implementation appears to be quite efficient when compared to TreeReduce and FullAxis implementations:

.. note::

    All of the examples below were run on Intel(R) Xeon(R) Gold 6238R CPU @ 2.20GHz (112 cores), 192gb RAM

.. code-block:: ipython

    In [4]: import modin.pandas as pd; import numpy as np

    In [5]: df = pd.DataFrame(np.random.randint(0, 1_000_000, size=(1_000_000, 10)), columns=[f"col{i}" for i in range(10)])

    In [6]: %timeit df.groupby("col0").nunique() # full-axis implementation
    Out[6]: # 2.73 s ± 28.8 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

    In [7]: import modin.config as cfg; cfg.RangePartitioning.put(True)

    In [8]: %timeit df.groupby("col0").nunique() # range-partitiong implementation
    Out[8]: # 595 ms ± 61.3 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

Although it may look like the range-partitioning implementation always outperforms the other ones, it's not actually true.
There's a decent overhead on building the range partitioning itself, meaning that the other implementations
may act better on smaller data sizes or when the grouping columns (a key column to build range partitioning)
have too few unique values (and thus fewer units of parallelization):

.. code-block:: ipython

    In [4]: import modin.pandas as pd; import numpy as np

    In [5]: df = pd.DataFrame({"col0": np.tile(list("abcde"), 50_000), "col1": np.arange(250_000)})

    In [6]: %timeit df.groupby("col0").sum() # TreeReduce implementation
    Out[6]: # 155 ms ± 5.02 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)

    In [7]: import modin.config as cfg; cfg.RangePartitioning.put(True)

    In [8]: %timeit df.groupby("col0").sum() # range-partitiong implementation
    Out[8]: # 230 ms ± 22.2 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

We're still looking for a heuristic that would be able to automatically switch to the best implementation
for each groupby case, but for now, we're offering to play with this switch on your own to see which
implementation works best for your particular case.

The range-partitioning groupby does not yet support all of the pandas API and falls back to an other
implementation with the respective warning if it meets an unsupported case:

.. code-block:: python

    In [14]: import modin.config as cfg; cfg.RangePartitioning.put(True)

    In [15]: df.groupby(level=0).sum()
    Out[15]: # UserWarning: Can't use range-partitiong groupby implementation because of:
        ...  # Range-partitioning groupby is only supported when grouping on a column(s) of the same frame.
        ...  # https://github.com/modin-project/modin/issues/5926
        ...  # Falling back to a TreeReduce implementation.

Range-partitioning Merge
""""""""""""""""""""""""

.. note::
    Range-partitioning approach is implemented only for "left" and "inner" merge and only
    when merging on a single column using `on` argument.

Range-partitioning merge replaces broadcast merge. It is recommended to use range-partitioning implementation
if the right dataframe in merge is as big as the left dataframe. In this case, range-partitioning
implementation works faster and consumes less RAM.

TODO: add perf measurements from https://github.com/modin-project/modin/pull/6966

'.unique()' and '.drop_duplicates()'
""""""""""""""""""""""""""""""""""""

Range-partitioning implementation of '.unique()'/'.drop_duplicates()' works best when the input data size is big (more than
5_000_000 rows) and when the output size is also expected to be big (no more than 80% values are duplicates).

TODO: add perf measurements from https://github.com/modin-project/modin/pull/7091

'.nunique()'
""""""""""""""""""""""""""""""""""""

.. note::

    Range-partitioning approach is implemented only for 'pd.Series.nunique()' and 1-column dataframes.
    For multi-column dataframes '.nunique()' can only use full-axis reduce implementation.

Range-partitioning implementation of '.nunique()'' works best when the input data size is big (more than
5_000_000 rows) and when the output size is also expected to be big (no more than 80% values are duplicates).

TODO: add perf measurements from https://github.com/modin-project/modin/pull/7101

Resample
""""""""

.. note::

    Range-partitioning approach doesn't support transform-like functions (like `.interpolate()`, `.ffill()`, `.bfill()`, ...)

It is recommended to use range-partitioning for resampling if you're dealing with a dataframe that has more than
5_000_000 rows and the expected output is also expected to be big (more than 500_000 rows).

TODO: add perf measurements from https://github.com/modin-project/modin/pull/7140

pivot_table
"""""""""""

Range-partitioning implementation is automatically applied for `df.pivot_table`
whenever possible, users can't control this.
