:orphan:

=========================
DataFrame Module Overview
=========================

.. currentmodule:: modin.pandas

Modin's ``pandas.DataFrame`` API
''''''''''''''''''''''''''''''''

Modin's ``pandas.DataFrame`` API is backed by a distributed object providing an identical
API to pandas. After the user calls some ``DataFrame`` function, this call is internally
rewritten into a representation that can be processed in parallel by the partitions. These
results can be e.g., reduced to single output, identical to the single threaded
pandas ``DataFrame`` method output.

..
    TODO: add link to the docs with detailed description of queries compilation
    and execution ater DOCS-#2996 is merged.

Usage Guide
'''''''''''

The most efficient way to create Modin ``DataFrame`` is to import data from external
storage using the highly efficient Modin IO methods (for example using ``pd.read_csv``,
see details for Modin IO methods in the :doc:`separate section </flow/modin/core/io/index>`),
but even if the data does not originate from a file, any pandas supported data type or
``pandas.DataFrame`` can be used. Internally, the ``DataFrame`` data is divided into
partitions, which number along an axis usually corresponds to the number of the user's hardware CPUs. If needed,
the number of partitions can be changed by setting ``modin.config.NPartitions``.

Let's consider simple example of creation and interacting with Modin ``DataFrame``:

.. code-block:: python

    import modin.config

    # This explicitly sets the number of partitions
    modin.config.NPartitions.put(4)

    import modin.pandas as pd
    import pandas

    # Create Modin DataFrame from the external file
    pd_dataframe = pd.read_csv("test_data.csv")
    # Create Modin DataFrame from the python object
    # data = {f'col{x}': [f'col{x}_{y}' for y in range(100, 356)] for x in range(4)}
    # pd_dataframe = pd.DataFrame(data)
    # Create Modin DataFrame from the pandas object
    # pd_dataframe = pd.DataFrame(pandas.DataFrame(data))

    # Show created DataFrame
    print(pd_dataframe)

    # List DataFrame partitions. Note, that internal API is intended for
    # developers needs and was used here for presentation purposes
    # only.
    partitions = pd_dataframe._query_compiler._modin_frame._partitions
    print(partitions)

    # Show the first DataFrame partition
    print(partitions[0][0].get())

    Output:

    # created DataFrame

            col0      col1      col2      col3
    0    col0_100  col1_100  col2_100  col3_100
    1    col0_101  col1_101  col2_101  col3_101
    2    col0_102  col1_102  col2_102  col3_102
    3    col0_103  col1_103  col2_103  col3_103
    4    col0_104  col1_104  col2_104  col3_104
    ..        ...       ...       ...       ...
    251  col0_351  col1_351  col2_351  col3_351
    252  col0_352  col1_352  col2_352  col3_352
    253  col0_353  col1_353  col2_353  col3_353
    254  col0_354  col1_354  col2_354  col3_354
    255  col0_355  col1_355  col2_355  col3_355

    [256 rows x 4 columns]

    # List of DataFrame partitions

    [[<modin.core.execution.ray.implementations.pandas_on_ray.partitioning.partition.PandasOnRayDataframePartition object at 0x7fc554e607f0>]
    [<modin.core.execution.ray.implementations.pandas_on_ray.partitioning.partition.PandasOnRayDataframePartition object at 0x7fc554e9a4f0>]
    [<modin.core.execution.ray.implementations.pandas_on_ray.partitioning.partition.PandasOnRayDataframePartition object at 0x7fc554e60820>]
    [<modin.core.execution.ray.implementations.pandas_on_ray.partitioning.partition.PandasOnRayDataframePartition object at 0x7fc554e609d0>]]

    # The first DataFrame partition
    
            col0      col1      col2      col3
    0   col0_100  col1_100  col2_100  col3_100
    1   col0_101  col1_101  col2_101  col3_101
    2   col0_102  col1_102  col2_102  col3_102
    3   col0_103  col1_103  col2_103  col3_103
    4   col0_104  col1_104  col2_104  col3_104
    ..       ...       ...       ...       ...
    60  col0_160  col1_160  col2_160  col3_160
    61  col0_161  col1_161  col2_161  col3_161
    62  col0_162  col1_162  col2_162  col3_162
    63  col0_163  col1_163  col2_163  col3_163
    64  col0_164  col1_164  col2_164  col3_164

    [65 rows x 4 columns]

As we show in the example above, Modin ``DataFrame`` can be easily created, and supports any input that pandas ``DataFrame`` supports.
Also note that tuning of the ``DataFrame`` partitioning can be done by just setting a single config.

Public API
----------

.. autosummary::
    :toctree: api/
    
    DataFrame
    DataFrame.abs
    DataFrame.add
    DataFrame.add_prefix
    DataFrame.add_suffix
    DataFrame.agg
    DataFrame.aggregate
    DataFrame.align
    DataFrame.all
    DataFrame.any
    DataFrame.append
    DataFrame.apply
    DataFrame.applymap
    DataFrame.asfreq
    DataFrame.asof
    DataFrame.assign
    DataFrame.astype
    DataFrame.at_time
    DataFrame.backfill
    DataFrame.between_time
    DataFrame.bfill
    DataFrame.bool
    DataFrame.boxplot
    DataFrame.clip
    DataFrame.combine
    DataFrame.combine_first
    DataFrame.compare
    DataFrame.convert_dtypes
    DataFrame.copy
    DataFrame.corr
    DataFrame.corrwith
    DataFrame.count
    DataFrame.cov
    DataFrame.cummax
    DataFrame.cummin
    DataFrame.cumprod
    DataFrame.cumsum
    DataFrame.describe
    DataFrame.diff
    DataFrame.div
    DataFrame.divide
    DataFrame.dot
    DataFrame.drop
    DataFrame.drop_duplicates
    DataFrame.droplevel
    DataFrame.dropna
    DataFrame.duplicated
    DataFrame.eq
    DataFrame.equals
    DataFrame.eval
    DataFrame.ewm
    DataFrame.expanding
    DataFrame.explode
    DataFrame.ffill
    DataFrame.fillna
    DataFrame.filter
    DataFrame.first
    DataFrame.first_valid_index
    DataFrame.floordiv
    DataFrame.from_dict
    DataFrame.from_records
    DataFrame.ge
    DataFrame.get
    DataFrame.groupby
    DataFrame.gt
    DataFrame.head
    DataFrame.hist
    DataFrame.idxmax
    DataFrame.idxmin
    DataFrame.infer_objects
    DataFrame.info
    DataFrame.insert
    DataFrame.interpolate
    DataFrame.isin
    DataFrame.isna
    DataFrame.isnull
    DataFrame.items
    DataFrame.iteritems
    DataFrame.iterrows
    DataFrame.itertuples
    DataFrame.join
    DataFrame.keys
    DataFrame.kurt
    DataFrame.kurtosis
    DataFrame.last
    DataFrame.last_valid_index
    DataFrame.le
    DataFrame.lookup
    DataFrame.lt
    DataFrame.mad
    DataFrame.mask
    DataFrame.max
    DataFrame.mean
    DataFrame.median
    DataFrame.melt
    DataFrame.memory_usage
    DataFrame.merge
    DataFrame.min
    DataFrame.mod
    DataFrame.mode
    DataFrame.mul
    DataFrame.multiply
    DataFrame.ne
    DataFrame.nlargest
    DataFrame.notna
    DataFrame.notnull
    DataFrame.nsmallest
    DataFrame.nunique
    DataFrame.pad
    DataFrame.pct_change
    DataFrame.pipe
    DataFrame.pivot
    DataFrame.pivot_table
    DataFrame.pop
    DataFrame.pow
    DataFrame.prod
    DataFrame.product
    DataFrame.quantile
    DataFrame.query
    DataFrame.radd
    DataFrame.rank
    DataFrame.rdiv
    DataFrame.reindex
    DataFrame.reindex_like
    DataFrame.rename
    DataFrame.rename_axis
    DataFrame.reorder_levels
    DataFrame.replace
    DataFrame.resample
    DataFrame.reset_index
    DataFrame.rfloordiv
    DataFrame.rmod
    DataFrame.rmul
    DataFrame.rolling
    DataFrame.round
    DataFrame.rpow
    DataFrame.rsub
    DataFrame.rtruediv
    DataFrame.sample
    DataFrame.select_dtypes
    DataFrame.sem
    DataFrame.set_axis
    DataFrame.set_flags
    DataFrame.set_index
    DataFrame.shift
    DataFrame.skew
    DataFrame.slice_shift
    DataFrame.sort_index
    DataFrame.sort_values
    DataFrame.squeeze
    DataFrame.stack
    DataFrame.std
    DataFrame.sub
    DataFrame.subtract
    DataFrame.sum
    DataFrame.swapaxes
    DataFrame.swaplevel
    DataFrame.tail
    DataFrame.take
    DataFrame.to_clipboard
    DataFrame.to_csv
    DataFrame.to_dict
    DataFrame.to_excel
    DataFrame.to_feather
    DataFrame.to_gbq
    DataFrame.to_hdf
    DataFrame.to_html
    DataFrame.to_json
    DataFrame.to_latex
    DataFrame.to_markdown
    DataFrame.to_numpy
    DataFrame.to_parquet
    DataFrame.to_period
    DataFrame.to_pickle
    DataFrame.to_pickle_distributed
    DataFrame.to_records
    DataFrame.to_sql
    DataFrame.to_stata
    DataFrame.to_string
    DataFrame.to_timestamp
    DataFrame.to_xarray
    DataFrame.to_xml
    DataFrame.transform
    DataFrame.transpose
    DataFrame.truediv
    DataFrame.truncate
    DataFrame.tshift
    DataFrame.tz_convert
    DataFrame.tz_localize
    DataFrame.unstack
    DataFrame.update
    DataFrame.value_counts
    DataFrame.var
    DataFrame.where
    DataFrame.xs

.. autosummary::
    :toctree: api/
    
    DataFrame.T
    DataFrame.at
    DataFrame.attrs
    DataFrame.axes
    DataFrame.columns
    DataFrame.dtypes
    DataFrame.empty
    DataFrame.flags
    DataFrame.iat
    DataFrame.iloc
    DataFrame.index
    DataFrame.loc
    DataFrame.ndim
    DataFrame.plot
    DataFrame.shape
    DataFrame.size
    DataFrame.style
    DataFrame.values
