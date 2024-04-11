Range-partitioning in Modin
###########################

Modin utilizes a range-partitioning approach for specific operations, significantly enhancing
parallelism and reducing memory consumption in certain scenarios. Range-partitioning is typically
engaged for operations that has key columns (to group on, to merge on, ...).

You can enable `range-partitioning`_ by specifying ``cfg.RangePartitioning`` :doc:`configuration variable: </flow/modin/config>`

.. code-block:: python

    import modin.pandas as pd
    import modin.config as cfg

    cfg.RangePartitioning.put(True) # past this point methods that support range-partitioning
                                    # will use engage it

    pd.DataFrame(...).groupby(...).mean() # use range-partitioning for groupby.mean()

    cfg.Range-partitioning.put(False)

    pd.DataFrame(...).groupby(...).mean() # use MapReduce implementation for groupby.mean()

Building range-partitioning assumes data reshuffling, which may result into breaking the original
order of rows, for some operation, it will mean that the result will be different from Pandas.

Range-partitioning is not a silver bullet, meaning that enabling it is not always beneficial. Below you find
a list of operations that have support for range-partitioning and practical advices on when one should
enable it.

GroupBy
=======

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

Merge
=====

.. note::
    Range-partitioning approach is implemented only for "left" and "inner" merge and only
    when merging on a single column using `on` argument.

Range-partitioning merge replaces broadcast merge. It is recommended to use range-partitioning implementation
if the right dataframe in merge is as big as the left dataframe. In this case, range-partitioning
implementation works faster and consumes less RAM.

Under the spoiler you can find performance comparison of range-partitioning and broadcast merge in different
scenarios:

.. raw:: html

   <details>
   <summary><a>Performance measurements for merge</a></summary>

The performance was measured on `h2o join queries`_ using Intel(R) Xeon(R) Gold 6238R CPU @ 2.20GHz (56 cores),
with the number of cores allocated for Modin limited by 44 (``MODIN_CPUS=44``).

Measurements for small 500mb data:

.. image:: /img/range_partitioning_measurements/merge_h2o_500mb.jpg
   :align: center

Measurements for medium 5gb data:

.. image:: /img/range_partitioning_measurements/merge_h2o_5gb.png
   :align: center

.. raw:: html

   </details>


``.unique()`` and ``.drop_duplicates()``
========================================

.. note::
    When range-partitioning is enabled, both ``.unique()`` and ``.drop_duplicates()`` will
    yield results that are sorted along rows. If range-partitioning is disabled,
    the original order will be maintained.

Range-partitioning implementation of ``.unique()`` / ``.drop_duplicates()`` works best when the input data size is big (more than
5_000_000 rows) and when the output size is also expected to be big (no more than 80% values are duplicates).

Under the spoiler you can find performance comparisons in different scenarios:

.. raw:: html

   <details>
   <summary><a>Performance measurements for ``.unique()``</a></summary>

The performance was measured on randomly generated data using Intel(R) Xeon(R) Gold 6238R CPU @ 2.20GHz (56 cores).
The `duplicate rate` shows the procentage. You can learn more about this micro-benchmark by reading its source code:

.. raw:: html

   <details>
   <summary><a>Micro-benchmark's source code</a></summary>

.. code-block:: python

    import modin.pandas as pd
    import numpy as np
    import modin.config as cfg

    from modin.utils import execute
    from timeit import default_timer as timer
    import pandas

    cfg.CpuCount.put(16)

    def get_data(nrows, dtype):
        if dtype == int:
            return np.arange(nrows)
        elif dtype == float:
            return np.arange(nrows).astype(float)
        elif dtype == str:
            return np.array([f"value{i}" for i in range(nrows)])
        else:
            raise NotImplementedError(dtype)

    pd.DataFrame(np.arange(cfg.NPartitions.get() * cfg.MinPartitionSize.get())).to_numpy()

    nrows = [1_000_000, 5_000_000, 10_000_000, 25_000_000, 50_000_000, 100_000_000]
    duplicate_rate = [0, 0.1, 0.5, 0.95]
    dtypes = [int, str]
    use_range_part = [True, False]

    columns = pandas.MultiIndex.from_product([dtypes, duplicate_rate, use_range_part], names=["dtype", "duplicate rate", "use range-part"])
    result = pandas.DataFrame(index=nrows, columns=columns)

    i = 0
    total_its = len(nrows) * len(duplicate_rate) * len(dtypes) * len(use_range_part)

    for dt in dtypes:
        for nrow in nrows:
            data = get_data(nrow, dt)
            np.random.shuffle(data)
            for dpr in duplicate_rate:
                data_c = data.copy()
                dupl_val = data_c[0]

                num_duplicates = int(dpr * nrow)
                dupl_indices = np.random.choice(np.arange(nrow), num_duplicates, replace=False)
                data_c[dupl_indices] = dupl_val

                for impl in use_range_part:
                    print(f"{round((i / total_its) * 100, 2)}%")
                    i += 1
                    cfg.RangePartitioning.put(impl)

                    sr = pd.Series(data_c)
                    execute(sr)

                    t1 = timer()
                    # returns a list, so no need for materialization
                    sr.unique()
                    tm = timer() - t1
                    print(nrow, dpr, dt, impl, tm)
                    result.loc[nrow, (dt, dpr, impl)] = tm
                    result.to_excel("unique.xlsx")

.. raw:: html

   </details>

Measurements with 16 cores being allocated for Modin (``MODIN_CPUS=16``):

.. image:: /img/range_partitioning_measurements/unique_16cpus.jpg
   :align: center

Measurements with 44 cores being allocated for Modin (``MODIN_CPUS=4``):

.. image:: /img/range_partitioning_measurements/unique_44cpus.jpg
   :align: center

.. raw:: html

   </details>


.. raw:: html

   <details>
   <summary><a>Performance measurements for ``.drop_duplicates()``</a></summary>

The performance was measured on randomly generated data using Intel(R) Xeon(R) Gold 6238R CPU @ 2.20GHz (56 cores).
The `duplicate rate` shows the procentage. The `subset size` shows the number of columns being specified as a ``subset``
parameter for ``df.drop_duplicates()``. You can learn more about this micro-benchmark by reading its source code:

.. raw:: html

   <details>
   <summary><a>Micro-benchmark's source code</a></summary>

.. code-block:: python

    import modin.pandas as pd
    import numpy as np
    import modin.config as cfg

    from modin.utils import execute
    from timeit import default_timer as timer
    import pandas

    cfg.CpuCount.put(16)

    pd.DataFrame(np.arange(cfg.NPartitions.get() * cfg.MinPartitionSize.get())).to_numpy()

    nrows = [1_000_000, 5_000_000, 10_000_000, 25_000_000]
    duplicate_rate = [0, 0.1, 0.5, 0.95]
    subset = [["col0"], ["col1", "col2", "col3", "col4"], None]
    ncols = 15
    use_range_part = [True, False]

    columns = pandas.MultiIndex.from_product(
        [
            [len(sbs) if sbs is not None else ncols for sbs in subset],
            duplicate_rate,
            use_range_part
        ],
        names=["subset size", "duplicate rate", "use range-part"]
    )
    result = pandas.DataFrame(index=nrows, columns=columns)

    i = 0
    total_its = len(nrows) * len(duplicate_rate) * len(subset) * len(use_range_part)

    for sbs in subset:
        for nrow in nrows:
            data = {f"col{i}": np.arange(nrow) for i in range(ncols)}
            pandas_df = pandas.DataFrame(data)

            for dpr in duplicate_rate:
                pandas_df_c = pandas_df.copy()
                dupl_val = pandas_df_c.iloc[0]

                num_duplicates = int(dpr * nrow)
                dupl_indices = np.random.choice(np.arange(nrow), num_duplicates, replace=False)
                pandas_df_c.iloc[dupl_indices] = dupl_val

                for impl in use_range_part:
                    print(f"{round((i / total_its) * 100, 2)}%")
                    i += 1
                    cfg.RangePartitioning.put(impl)

                    md_df = pd.DataFrame(pandas_df_c)
                    execute(md_df)

                    t1 = timer()
                    res = md_df.drop_duplicates(subset=sbs)
                    execute(res)
                    tm = timer() - t1

                    sbs_s = len(sbs) if sbs is not None else ncols
                    print("len()", res.shape, nrow, dpr, sbs_s, impl, tm)
                    result.loc[nrow, (sbs_s, dpr, impl)] = tm
                    result.to_excel("drop_dupl.xlsx")

.. raw:: html

   </details>

Measurements with 16 cores being allocated for Modin (``MODIN_CPUS=16``):

.. image:: /img/range_partitioning_measurements/drop_duplicates_16cpus.jpg
   :align: center

Measurements with 44 cores being allocated for Modin (``MODIN_CPUS=44``):

.. image:: /img/range_partitioning_measurements/drop_duplicates_44cpus.jpg
   :align: center

.. raw:: html

   </details>


'.nunique()'
============

.. note::

    Range-partitioning approach is implemented only for ``pd.Series.nunique()`` and 1-column dataframes.
    For multi-column dataframes ``.nunique()`` can only use full-axis reduce implementation.

Range-partitioning implementation of '.nunique()'' works best when the input data size is big (more than
5_000_000 rows) and when the output size is also expected to be big (no more than 80% values are duplicates).

Under the spoiler you can find performance comparisons in different scenarios:

.. raw:: html

   <details>
   <summary><a>Performance measurements for ``.nunique()``</a></summary>

The performance was measured on randomly generated data using Intel(R) Xeon(R) Gold 6238R CPU @ 2.20GHz (56 cores).
The `duplicate rate` shows the procentage. You can learn more about this micro-benchmark by reading its source code:

.. raw:: html

   <details>
   <summary><a>Micro-benchmark's source code</a></summary>

.. code-block:: python

    import modin.pandas as pd
    import numpy as np
    import modin.config as cfg

    from modin.utils import execute
    from timeit import default_timer as timer
    import pandas

    cfg.CpuCount.put(16)

    def get_data(nrows, dtype):
        if dtype == int:
            return np.arange(nrows)
        elif dtype == float:
            return np.arange(nrows).astype(float)
        elif dtype == str:
            return np.array([f"value{i}" for i in range(nrows)])
        else:
            raise NotImplementedError(dtype)

    pd.DataFrame(np.arange(cfg.NPartitions.get() * cfg.MinPartitionSize.get())).to_numpy()

    nrows = [1_000_000, 5_000_000, 10_000_000, 25_000_000, 50_000_000, 100_000_000]
    duplicate_rate = [0, 0.1, 0.5, 0.95]
    dtypes = [int, str]
    use_range_part = [True, False]

    columns = pandas.MultiIndex.from_product([dtypes, duplicate_rate, use_range_part], names=["dtype", "duplicate rate", "use range-part"])
    result = pandas.DataFrame(index=nrows, columns=columns)

    i = 0
    total_its = len(nrows) * len(duplicate_rate) * len(dtypes) * len(use_range_part)

    for dt in dtypes:
        for nrow in nrows:
            data = get_data(nrow, dt)
            np.random.shuffle(data)
            for dpr in duplicate_rate:
                data_c = data.copy()
                dupl_val = data_c[0]

                num_duplicates = int(dpr * nrow)
                dupl_indices = np.random.choice(np.arange(nrow), num_duplicates, replace=False)
                data_c[dupl_indices] = dupl_val

                for impl in use_range_part:
                    print(f"{round((i / total_its) * 100, 2)}%")
                    i += 1
                    cfg.RangePartitioning.put(impl)

                    sr = pd.Series(data_c)
                    execute(sr)

                    t1 = timer()
                    # returns a scalar, so no need for materialization
                    res = sr.nunique()
                    tm = timer() - t1
                    print(nrow, dpr, dt, impl, tm)
                    result.loc[nrow, (dt, dpr, impl)] = tm
                    result.to_excel("nunique.xlsx")

.. raw:: html

   </details>

Measurements with 16 cores being allocated for Modin (``MODIN_CPUS=16``):

.. image:: /img/range_partitioning_measurements/nunique_16cpus.jpg
   :align: center


.. raw:: html

   </details>

Resample
========

.. note::

    Range-partitioning approach doesn't support transform-like functions (like `.interpolate()`, `.ffill()`, `.bfill()`, ...)

It is recommended to use range-partitioning for resampling if you're dealing with a dataframe that has more than
5_000_000 rows and the expected output is also expected to be big (more than 500_000 rows).

Under the spoiler you can find performance comparisons in different scenarios:

.. raw:: html

   <details>
   <summary><a>Performance measurements for ``.resample()``</a></summary>

The script below measures performance of ``df.resample(rule).sum()`` using Intel(R) Xeon(R) Gold 6238R CPU @ 2.20GHz (56 cores).
You can learn more about this micro-benchmark by reading its source code:

.. raw:: html

   <details>
   <summary><a>Micro-benchmark's source code</a></summary>

.. code-block:: python

    import pandas
    import numpy as np
    import modin.pandas as pd
    import modin.config as cfg

    from timeit import default_timer as timer

    from modin.utils import execute

    cfg.CpuCount.put(16)

    nrows = [1_000_000, 5_000_000, 10_000_000]
    ncols = [5, 33]
    rules = [
        "500ms", # doubles nrows
        "30s", # decreases nrows in 30 times
        "5min", # decreases nrows in 300
    ]
    use_rparts = [True, False]

    cols = pandas.MultiIndex.from_product([rules, ncols, use_rparts], names=["rule", "ncols", "USE RANGE PART"])
    rres = pandas.DataFrame(index=nrows, columns=cols)

    total_nits = len(nrows) * len(ncols) * len(rules) * len(use_rparts)
    i = 0

    for nrow in nrows:
        for ncol in ncols:
            index = pandas.date_range("31/12/2000", periods=nrow, freq="s")
            data = {f"col{i}": np.arange(nrow) for i in range(ncol)}
            pd_df = pandas.DataFrame(data, index=index)
            for rule in rules:
                for rparts in use_rparts:
                    print(f"{round((i / total_nits) * 100, 2)}%")
                    i += 1
                    cfg.RangePartitioning.put(rparts)

                    df = pd.DataFrame(data, index=index)
                    execute(df)

                    t1 = timer()
                    res = df.resample(rule).sum()
                    execute(res)
                    ts = timer() - t1
                    print(nrow, ncol, rule, rparts, ts)

                    rres.loc[nrow, (rule, ncol, rparts)] = ts
                    rres.to_excel("resample.xlsx")

.. raw:: html

   </details>

Measurements with 16 cores being allocated for Modin (``MODIN_CPUS=16``):

.. image:: /img/range_partitioning_measurements/resample_16cpus.jpg
   :align: center


.. raw:: html

   </details>

pivot_table
===========

Range-partitioning implementation is automatically applied for ``df.pivot_table``
whenever possible, users can't control this.


.. _h2o join queries: https://h2oai.github.io/db-benchmark/
.. _range-partitioning: https://www.techopedia.com/definition/31994/range-partitioning
