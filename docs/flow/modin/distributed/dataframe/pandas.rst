Pandas partitioning API
=======================

This page contains a description of the API to extract partitions from and build Modin Dataframes.

unwrap_partitions
-----------------

.. autofunction:: modin.distributed.dataframe.pandas.unwrap_partitions

from_partitions
---------------
.. autofunction:: modin.distributed.dataframe.pandas.from_partitions

Example
-------

.. code-block:: python

  import modin.pandas as pd
  from modin.distributed.dataframe.pandas import unwrap_partitions, from_partitions
  import numpy as np
  data = np.random.randint(0, 100, size=(2 ** 10, 2 ** 8))
  df = pd.DataFrame(data)
  partitions = unwrap_partitions(df, axis=0, get_ip=True)
  print(partitions)
  new_df = from_partitions(partitions, axis=0)
  print(new_df)
