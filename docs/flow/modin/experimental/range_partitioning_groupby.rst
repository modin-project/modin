Range-partitioning GroupBy
""""""""""""""""""""""""""

The range-partitioning GroupBy implementation utilizes Modin's reshuffling mechanism that gives an
ability to build range partitioning over a Modin DataFrame.

In order to enable/disable the range-partitiong implementation you have to specify ``cfg.RangePartitioningGroupby``
:doc:`configuration variable: </flow/modin/config>`

.. code-block:: ipython

    In [4]: import modin.config as cfg; cfg.RangePartitioningGroupby.put(True)

    In [5]: # past this point, Modin will always use the range-partitiong groupby implementation

    In [6]: cfg.RangePartitioningGroupby.put(False)

    In [7]: # past this point, Modin won't use range-partitiong groupby implementation anymore

The range-partitiong implementation appears to be quite efficient when compared to TreeReduce and FullAxis implementations:

.. note::

    All of the examples below were run on Intel(R) Xeon(R) Gold 6238R CPU @ 2.20GHz (112 cores), 192gb RAM

.. code-block:: ipython

    In [4]: import modin.pandas as pd; import numpy as np

    In [5]: df = pd.DataFrame(np.random.randint(0, 1_000_000, size=(1_000_000, 10)), columns=[f"col{i}" for i in range(10)])

    In [6]: %timeit df.groupby("col0").nunique() # full-axis implementation
    Out[6]: # 2.73 s ± 28.8 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

    In [7]: import modin.config as cfg; cfg.RangePartitioningGroupby.put(True)

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

    In [7]: import modin.config as cfg; cfg.RangePartitioningGroupby.put(True)

    In [8]: %timeit df.groupby("col0").sum() # range-partitiong implementation
    Out[8]: # 230 ms ± 22.2 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

We're still looking for a heuristic that would be able to automatically switch to the best implementation
for each groupby case, but for now, we're offering to play with this switch on your own to see which
implementation works best for your particular case.

The range-partitioning groupby does not yet support all of the pandas API and falls back to an other
implementation with the respective warning if it meets an unsupported case:

.. code-block:: python

    In [14]: import modin.config as cfg; cfg.RangePartitioningGroupby.put(True)

    In [15]: df.groupby(level=0).sum()
    Out[15]: # UserWarning: Can't use range-partitiong groupby implementation because of:
        ...  # Range-partitioning groupby is only supported when grouping on a column(s) of the same frame.
        ...  # https://github.com/modin-project/modin/issues/5926
        ...  # Falling back to a TreeReduce implementation.

Range-partitioning Merge
""""""""""""""""""""""""

It is recommended to use this implementation if the right dataframe in merge is as big as
the left dataframe. In this case, range-partitioning implementation works faster and consumes less RAM.
