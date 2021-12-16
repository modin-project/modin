Optimization Guide
==================

Here you can find information on Modin optimizations both for general query pipeline and for specific operations.

Generic settings
""""""""""""""""

If you are working with a wide DataFrame and performing an operation that requires ``map`` Core Dataframe Algebra operator
to be applied cell-wise, you may want to change the number of DataFrame partitions to avoid
an excessive amount of remote partitions, which is much greater than the number of CPUs.

.. code-block:: python

  from multiprocessing import cpu_count
  import modin.config as cfg
  import modin.pandas as pd
  import numpy as np

  data = np.random.randint(0, 100, size=(2**15, 2**15))

  cfg.NPartitions.put(cpu_count())
  df = pd.DataFrame(data)
  %timeit df.abs()
  1.9 s  49.8 ms per loop (mean  std. dev. of 7 runs, 1 loop each)

  cfg.NPartitions.put(int(cpu_count() / 2))
  df = pd.DataFrame(data)
  %timeit df.abs()
  506 ms  17.5 ms per loop (mean  std. dev. of 7 runs, 1 loop each)

Operation-specific settings
"""""""""""""""""""""""""""

merge
-----

``merge`` operation in Modin uses the broadcast join algorithm: combining a right Modin DataFrame into a pandas DataFrame and
broadcasting it to the row partitions of the left Modin DataFrame. In order to minimize interprocess communication cost when doing
an inner join you may want to swap left and right DataFrames.

.. code-block:: python

  import modin.pandas as pd
  import numpy as np

  left_data = np.random.randint(0, 100, size=(2**8, 2**8))
  right_data = np.random.randint(0, 100, size=(2**12, 2**12))

  left_df = pd.DataFrame(left_data)
  right_df = pd.DataFrame(right_data)
  %timeit left_df.merge(right_df, how="inner", on=10)
  3.59 s  107 ms per loop (mean  std. dev. of 7 runs, 1 loop each)

  %timeit right_df.merge(left_df, how="inner", on=10)
  1.22 s  40.1 ms per loop (mean  std. dev. of 7 runs, 1 loop each)

Note that result columns order may differ for first and second ``merge``.
