Out of Core in Modin
====================

If you are working with very large files or would like to exceed your memory, you may
change the primary location of the `DataFrame`_. If you would like to exceed memory, you
can use your disk as an overflow for the memory.

Starting Modin with out of core enabled
---------------------------------------

Out of core is now enabled by default for both Ray and Dask engines.

Disabling Out of Core
---------------------

Out of core is enabled by the compute engine selected. To disable it, start your
preferred compute engine with the appropriate arguments. For example:

.. code-block:: python

  import modin.pandas as pd
  import ray

  ray.init(_plasma_directory="/tmp")  # setting to disable out of core in Ray
  df = pd.read_csv("some.csv")

Running an example with out of core
-----------------------------------

Before you run this, please make sure you follow the instructions listed above.

.. code-block:: python

  import modin.pandas as pd
  import numpy as np
  frame_data = np.random.randint(0, 100, size=(2**20, 2**8)) # 2GB each
  df = pd.DataFrame(frame_data).add_prefix("col")
  big_df = pd.concat([df for _ in range(20)]) # 20x2GB frames
  print(big_df)
  nan_big_df = big_df.isna() # The performance here represents a simple map
  print(big_df.groupby("col1").count()) # group by on a large dataframe

This example creates a 40GB DataFrame from 20 identical 2GB DataFrames and performs
various operations on them. Feel free to play around with this code and let us know what
you think!

.. _Dataframe: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html
