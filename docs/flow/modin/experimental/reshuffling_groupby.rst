Reshuffling GroupBy
"""""""""""""""""""

The experimental GroupBy implementation utilizes Modin's reshuffling mechanism that gives an
ability to build range partitioning over a Modin DataFrame.

In order to enable/disable this new implementation you have to specify ``cfg.ExperimentalGroupbyImpl``
:doc:`configuration variable: </flow/modin/config>`

.. code-block:: ipython

    In [4]: import modin.config as cfg; cfg.ExperimentalGroupbyImpl.put(True)

    In [5]: # past this point, Modin will always use the new reshuffling groupby implementation

    In [6]: cfg.ExperimentalGroupbyImpl.put(False)

    In [7]: # past this point, Modin won't use reshuffling groupby implementation anymore

The reshuffling implementation appears to be quite efficient when compared to old TreeReduce and FullAxis implementations:

.. note::

    All of the examples below were run on Intel(R) Xeon(R) Gold 6238R CPU @ 2.20GHz (112 cores), 192gb RAM

.. code-block:: ipython

    In [4]: import modin.pandas as pd; import numpy as np

    In [5]: df = pd.DataFrame(np.random.randint(0, 1_000_000, size=(1_000_000, 10)), columns=[f"col{i}" for i in range(10)])

    In [6]: %timeit df.groupby("col0").nunique() # old full-axis implementation
    Out[6]: # 2.73 s ± 28.8 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

    In [7]: import modin.config as cfg; cfg.ExperimentalGroupbyImpl.put(True)

    In [8]: %timeit df.groupby("col0").nunique() # new reshuffling implementation
    Out[8]: # 595 ms ± 61.3 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

Although it may look like the new implementation always outperforms the old ones, it's not actually true.
There's a decent overhead on building the range partitioning itself, meaning that the old implementations
may act better on smaller data sizes or when the grouping columns (a key column to build range partitioning)
have too few unique values (and thus fewer units of parallelization):

.. code-block:: ipython

    In [4]: import modin.pandas as pd; import numpy as np

    In [5]: df = pd.DataFrame({"col0": np.tile(list("abcde"), 50_000), "col1": np.arange(250_000)})

    In [6]: %timeit df.groupby("col0").sum() # old TreeReduce implementation
    Out[6]: # 155 ms ± 5.02 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)

    In [7]: import modin.config as cfg; cfg.ExperimentalGroupbyImpl.put(True)

    In [8]: %timeit df.groupby("col0").sum() # new reshuffling implementation
    Out[8]: # 230 ms ± 22.2 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

We're still looking for a heuristic that would be able to automatically switch to the best implementation
for each groupby case, but for now, we're offering to play with this switch on your own to see which
implementation works best for your particular case.

The new experimental groupby does not yet support all of the pandas API and falls back to older
implementation with the respective warning if it meets an unsupported case:

.. code-block:: python

    In [14]: import modin.config as cfg; cfg.ExperimentalGroupbyImpl.put(True)

    In [15]: df.groupby(level=0).sum()
    Out[15]: # UserWarning: Can't use experimental reshuffling groupby implementation because of: 
        ...  # Reshuffling groupby is only supported when grouping on a column(s) of the same frame.
        ...  # https://github.com/modin-project/modin/issues/5926
        ...  # Falling back to a TreeReduce implementation.
