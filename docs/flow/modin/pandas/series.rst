:orphan:

======================
Series Module Overview
======================

.. currentmodule:: modin.pandas

Modin's ``pandas.Series`` API
'''''''''''''''''''''''''''''

Modin's ``pandas.Series`` API is backed by a distributed object providing an identical
API to pandas. After the user calls some ``Series`` function, this call is internally rewritten
into a representation that can be processed in parallel by the partitions. These
results can be e.g., reduced to single output, identical to the single threaded
pandas ``Series`` method output.

..
    TODO: add link to the docs with detailed description of queries compilation
    and execution ater DOCS-#2996 is merged.

Usage Guide
'''''''''''

The most efficient way to create Modin ``Series`` is to import data from external
storage using the highly efficient Modin IO methods (for example using ``pd.read_csv``,
see details for Modin IO methods in the :doc:`separate section </flow/modin/core/io/index>`),
but even if the data does not originate from a file, any pandas supported data type or
``pandas.Series`` can be used. Internally, the ``Series`` data is divided into
partitions, which number along an axis usually corresponds to the number of the user's hardware CPUs. If needed,
the number of partitions can be changed by setting ``modin.config.NPartitions``.

Let's consider simple example of creation and interacting with Modin ``Series``:

.. code-block:: python

    import modin.config

    # This explicitly sets the number of partitions
    modin.config.NPartitions.put(4)

    import modin.pandas as pd
    import pandas

    # Create Modin Series from the external file
    pd_series = pd.read_csv("test_data.csv", header=None).squeeze()
    # Create Modin Series from the python object
    # pd_series = pd.Series([x for x in range(256)])
    # Create Modin Series from the pandas object
    # pd_series = pd.Series(pandas.Series([x for x in range(256)]))

    # Show created `Series`
    print(pd_series)

    # List `Series` partitions. Note, that internal API is intended for
    # developers needs and was used here for presentation purposes
    # only.
    partitions = pd_series._query_compiler._modin_frame._partitions
    print(partitions)

    # Show the first `Series` partition
    print(partitions[0][0].get())

    Output:

    # created `Series`

    0      100
    1      101
    2      102
    3      103
    4      104
        ...
    251    351
    252    352
    253    353
    254    354
    255    355
    Name: 0, Length: 256, dtype: int64

    # List of `Series` partitions

    [[<modin.core.execution.ray.implementations.pandas_on_ray.partitioning.partition.PandasOnRayDataframePartition object at 0x7fc554e607f0>]
    [<modin.core.execution.ray.implementations.pandas_on_ray.partitioning.partition.PandasOnRayDataframePartition object at 0x7fc554e9a4f0>]
    [<modin.core.execution.ray.implementations.pandas_on_ray.partitioning.partition.PandasOnRayDataframePartition object at 0x7fc554e60820>]
    [<modin.core.execution.ray.implementations.pandas_on_ray.partitioning.partition.PandasOnRayDataframePartition object at 0x7fc554e609d0>]]

    # The first `Series` partition
    
        0
    0   100
    1   101
    2   102
    3   103
    4   104
    ..  ...
    60  160
    61  161
    62  162
    63  163
    64  164

    [65 rows x 1 columns]

As we show in the example above, Modin ``Series`` can be easily created, and supports any input that pandas ``Series`` supports.
Also note that tuning of the ``Series`` partitioning can be done by just setting a single config.

Public API
----------

.. autosummary::
    :toctree: api/
    
    Series
    Series.abs
    Series.add
    Series.add_prefix
    Series.add_suffix
    Series.agg
    Series.aggregate
    Series.align
    Series.all
    Series.any
    Series.append
    Series.apply
    Series.argmax
    Series.argmin
    Series.argsort
    Series.asfreq
    Series.asof
    Series.astype
    Series.at_time
    Series.autocorr
    Series.backfill
    Series.between
    Series.between_time
    Series.bfill
    Series.bool
    Series.clip
    Series.combine
    Series.combine_first
    Series.compare
    Series.convert_dtypes
    Series.copy
    Series.corr
    Series.count
    Series.cov
    Series.cummax
    Series.cummin
    Series.cumprod
    Series.cumsum
    Series.describe
    Series.diff
    Series.div
    Series.divide
    Series.divmod
    Series.dot
    Series.drop
    Series.drop_duplicates
    Series.droplevel
    Series.dropna
    Series.duplicated
    Series.eq
    Series.equals
    Series.ewm
    Series.expanding
    Series.explode
    Series.factorize
    Series.ffill
    Series.fillna
    Series.filter
    Series.first
    Series.first_valid_index
    Series.floordiv
    Series.ge
    Series.get
    Series.groupby
    Series.gt
    Series.head
    Series.hist
    Series.idxmax
    Series.idxmin
    Series.infer_objects
    Series.info
    Series.interpolate
    Series.isin
    Series.isna
    Series.isnull
    Series.item
    Series.items
    Series.iteritems
    Series.keys
    Series.kurt
    Series.kurtosis
    Series.last
    Series.last_valid_index
    Series.le
    Series.lt
    Series.mad
    Series.map
    Series.mask
    Series.max
    Series.mean
    Series.median
    Series.memory_usage
    Series.min
    Series.mod
    Series.mode
    Series.mul
    Series.multiply
    Series.ne
    Series.nlargest
    Series.notna
    Series.notnull
    Series.nsmallest
    Series.nunique
    Series.pad
    Series.pct_change
    Series.pipe
    Series.pop
    Series.pow
    Series.prod
    Series.product
    Series.quantile
    Series.radd
    Series.rank
    Series.ravel
    Series.rdiv
    Series.rdivmod
    Series.reindex
    Series.reindex_like
    Series.rename
    Series.rename_axis
    Series.reorder_levels
    Series.repeat
    Series.replace
    Series.resample
    Series.reset_index
    Series.rfloordiv
    Series.rmod
    Series.rmul
    Series.rolling
    Series.round
    Series.rpow
    Series.rsub
    Series.rtruediv
    Series.sample
    Series.searchsorted
    Series.sem
    Series.set_axis
    Series.set_flags
    Series.shift
    Series.skew
    Series.slice_shift
    Series.sort_index
    Series.sort_values
    Series.squeeze
    Series.std
    Series.sub
    Series.subtract
    Series.sum
    Series.swapaxes
    Series.swaplevel
    Series.tail
    Series.take
    Series.to_clipboard
    Series.to_csv
    Series.to_dict
    Series.to_excel
    Series.to_frame
    Series.to_hdf
    Series.to_json
    Series.to_latex
    Series.to_list
    Series.to_markdown
    Series.to_numpy
    Series.to_period
    Series.to_pickle
    Series.to_sql
    Series.to_string
    Series.to_timestamp
    Series.to_xarray
    Series.tolist
    Series.transform
    Series.transpose
    Series.truediv
    Series.truncate
    Series.tshift
    Series.tz_convert
    Series.tz_localize
    Series.unique
    Series.unstack
    Series.update
    Series.value_counts
    Series.var
    Series.view
    Series.where
    Series.xs

.. autosummary::
    :toctree: api/
    
    Series.T
    Series.array
    Series.at
    Series.attrs
    Series.axes
    Series.cat
    Series.dt
    Series.dtype
    Series.dtypes
    Series.empty
    Series.flags
    Series.hasnans
    Series.iat
    Series.iloc
    Series.index
    Series.is_monotonic
    Series.is_monotonic_decreasing
    Series.is_monotonic_increasing
    Series.is_unique
    Series.loc
    Series.name
    Series.nbytes
    Series.ndim
    Series.plot
    Series.shape
    Series.size
    Series.str
    Series.values
