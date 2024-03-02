:orphan:

Series Module Overview
""""""""""""""""""""""

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

Public API
----------

.. autoclass:: modin.pandas.series.Series

Usage Guide
'''''''''''

The most efficient way to create Modin ``Series`` is to import data from external
storage using the highly efficient Modin IO methods (for example using ``pd.read_csv``,
see details for Modin IO methods in the :doc:`IO </flow/modin/core/io/index>` page),
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
