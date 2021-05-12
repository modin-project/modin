:orphan:

Series Module Overview
""""""""""""""""""""""

Modin Series
''''''''''''
Modin Series represents disrtibuted `pandas.Series` object providing the same
pandas API. Internally passed data is divided into partitions in order to
parallelize computations and utilize the user's hardware as much as possible.
After the user calls some `Series` function, this call is converted into the query,
that can be processed in parallel by the partitions and then these results
reduced to single output, that is similar to single threaded pandas `Series` method
output.

..
    TODO: add link to the docs with detailed description of queries compilation
    and execution ater DOCS-#2996 is merged.

Usage Guide
'''''''''''
The most efficient way to create Modin `Series` is to import it's data from the external
storage using high efficient Modin IO methods (for example using `pd.read_csv`, see details for
Modin IO methods in the :doc:`separate section </flow/modin/engines/base/io>`), but even if
you don't have such storage, any pandas supported data type or `pandas.Series` themself
can be used. Most of the times `Series` data is evenly distributed across all partitions,
which number corresponds to the number of the user's hardware CPUs, but if it is needed
the number of partitions can be changed by seeting Modin config.

Let's consider simple example of creation and interacting with Modin `Series`:

.. code-block:: python

    import os

    # This explicitely sets the number of partitions
    os.environ["MODIN_CPUS"] = "4"

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

    [[<modin.engines.ray.pandas_on_ray.frame.partition.PandasOnRayFramePartition object at 0x000001E7CD11BD60>]
    [<modin.engines.ray.pandas_on_ray.frame.partition.PandasOnRayFramePartition object at 0x000001E7CD11BE50>]
    [<modin.engines.ray.pandas_on_ray.frame.partition.PandasOnRayFramePartition object at 0x000001E7CD11BF40>]
    [<modin.engines.ray.pandas_on_ray.frame.partition.PandasOnRayFramePartition object at 0x000001E7CD13E070>]]

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

As it can be seen from the example above, Modin `Series` can be easily created similarly to pandas `Series`
and tuning of it's partitioning can be done by setting of the single config.