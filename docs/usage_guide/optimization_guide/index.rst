Optimization Guide
==================

Here you can find more information about Modin's optimizations both for a pipeline of operations as well as for specific operations.


Understanding Modin's partitioning mechanism
"""""""""""""""""""""""""""""""""""""""""""""""
Modin's partitioning is crucial for performance; so we recommend expert users to understand Modin's
partitioning mechanism and how to tune it, to achieve better performance.

How does Modin partition a dataframe
------------------------------------

Modin uses a partitioning scheme that partitions a dataframe along both axes, resulting in a matrix
of partitions. The row and column chunk sizes are computed independently based
on the length of the appropriate axis and Modin's special :doc:`configuration variables </flow/modin/config>`
(``NPartitions`` and ``MinPartitionSize``):

- `NPartitions` is the maximum number of splits along an axis; by default, it equals the number of cores on the machine.
- `MinPartitionSize` is the minimum number of rows/columns to do a split. For instance, if `MinPartitionSize`
  is 32, the column axis will not be split unless the amount of columns is greater than 32. If it is is greater, for example, 34, 
  then the column axis is sliced into two partitions: containing 32 and 2 columns accordingly.

Beware that ``NPartitions`` specifies a limit for the number of partitions `along a single axis`, which means, that
the actual limit for the entire dataframe itself is the square of ``NPartitions``.

.. figure:: /img/partitioning_mechanism/partitioning_examples.svg
   :align: center

Full-axis functions
-------------------

Some of the aggregation functions require knowledge about the whole axis, for example at ``.apply(foo, axis=0)``
the passed function ``foo`` expects to receive data for the whole column at once.

When a full-axis function is applied, the partitions along this axis are being collected to a single worker
that processes the function. After the function is done, the data is split again across all of the workers.

.. figure:: /img/partitioning_mechanism/full_axis_function.svg
   :align: center

Note that the amount of remote calls is equal to the number of partitions, which means that since the number
of partitions is decreased for full-axis functions it also decreases its parallelism potential.

Also note, that the reduction functions such as ``.sum()``, ``.mean()``, ``.max()``, etc are not considered
to be full-axis and so not suffering from the decreasing of parallelism.

How to tune partitioning
------------------------

As you can see from the examples above, the more the frame's shape is closer to a square, the closer the number of
partitions to the square of `NPartitions`. In the case of `NPartitions` equals to the number of workers,
that means that a single worker is going to process multiple partitions at once, which slows down overall performance.

If your workflow mainly operates with wide frames and non-full-axis functions, it makes sense to reduce the
amount of `NPartitions` so a single worker would process a single partition.

.. figure:: /img/partitioning_mechanism/repartition_square_frames.svg
   :align: center

Copy-pastable example, showing how tuning ``NPartition`` value for wide frames may improve performance on your machine:

.. code-block:: python

  from multiprocessing import cpu_count
  import modin.config as cfg
  import modin.pandas as pd
  import numpy as np
  import timeit

  # Generating data for a square-like frame
  data = np.random.randint(0, 100, size=(2**10, 2**10))

  # Explicitly setting `NPartitions` to its default value
  cfg.NPartitions.put(cpu_count())

  # Each worker process `sqrt(cpu_count())` amount of partitions
  df = pd.DataFrame(data)
  print(f"10 times of .abs(): {timeit.timeit(lambda: df.abs(), number=10)}s.")
  # Possible output: 2.59s.

  # Taking a square root of the the current `NPartitions` to make more even partitioning
  cfg.NPartitions.put(int(cpu_count() ** 0.5))

  # Each worker process a single partition
  df = pd.DataFrame(data)
  print(f"10 times of .abs(): {timeit.timeit(lambda: df.abs(), number=10)}s.")
  # Possible output: 0.24s.

Do not iterate over Modin DataFrame
"""""""""""""""""""""""""""""""""""

Use ``df.apply()`` or other aggregation methods when possible instead of iterating over a frame.
For-loops don't scale and force to collect all the distributed data to the driver.

Copy-pastable example, showing how replacing a for-loop to the equivalent ``.apply()`` may improve performance:

.. code-block:: python

  import modin.pandas as pd
  import numpy as np
  from timeit import default_timer as timer

  data = np.random.randint(1, 100, (2 ** 10, 2 ** 2))

  md_df = pd.DataFrame(data)

  result = []
  t1 = timer()
  # Iterating over a frame forces to collect distributed data to the driver and doesn't scale
  for idx, row in md_df.iterrows():
      result.append((row[1] + row[2]) / row[3])
  print(f"Filling a list by iterating a Modin frame: {timer() - t1:.2f}s.")
  # Possible output: 36.15s.

  t1 = timer()
  # Using `.apply()` perfectly scales to all axis-partitions
  result = md_df.apply(lambda row: (row[1] + row[2]) / row[3], axis=1).to_numpy().tolist()
  print(f"Filling a list by using '.apply()' and converting the result to a list: {timer() - t1:.2f}s.")
  # Possible output: 0.22s.

Use Modin's algebra API to implement custom parallel functions
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

Modin provides a set of low-level parallel-implemented operators which can be used to build most of the
aggregation functions. These operators are present in the :doc:`algebra module </flow/modin/core/dataframe/algebra>`.
Modin DataFrame allows for users to use their own aggregations built with this module. Visit the 
:doc:`appropriate section </flow/modin/core/dataframe/algebra>` of the documentation for the steps to do it.

Do not mix pandas and Modin DataFrames
""""""""""""""""""""""""""""""""""""""

Although Modin is considered to be a drop-in replacement for pandas, they are not intended to be used together
in a single flow. Passing a pandas DataFrame as an argument for a Modin's frame method may either slowdown
the function (because it has to process non-distributed object) or raise an error. You would also get an undefined
behaviour if pass a Modin frame to pandas methods, pandas identifies Modin's objects as a simple iterable,
and so can't use its potential.

Copy-pastable example, showing how mixing pandas and Modin frames in a single flow may bottleneck performance:

.. code-block:: python

  import modin.pandas as pd
  import numpy as np
  import timeit
  import pandas

  data = np.random.randint(0, 100, (2 ** 20, 2 ** 2))

  md_df, md_df_copy = pd.DataFrame(data), pd.DataFrame(data)
  pd_df, pd_df_copy = pandas.DataFrame(data), pandas.DataFrame(data)

  print("concat modin frame + pandas frame:")
  # Concatenating modin frame + pandas frame using modin '.concat()'
  # This case is bad because Modin have to process non-distributed pandas object
  time = timeit.timeit(lambda: pd.concat([md_df, pd_df]), number=10)
  print(f"\t{time}s.\n")
  # Possible output: 0.44s.

  print("concat modin frame + modin frame:")
  # Concatenating modin frame + modin frame using modin '.concat()'
  # This is an ideal case, Modin is being used as intended
  time = timeit.timeit(lambda: pd.concat([md_df, md_df_copy]), number=10)
  print(f"\t{time}s.\n")
  # Possible output: 0.05s.

  print("concat pandas frame + pandas frame:")
  # Concatenating pandas frame + pandas frame using pandas '.concat()'
  time = timeit.timeit(lambda: pandas.concat([pd_df, pd_df_copy]), number=10)
  print(f"\t{time}s.\n")
  # Possible output: 0.31s.

  print("concat pandas frame + modin frame:")
  # Concatenating pandas frame + modin frame using pandas '.concat()'
  time = timeit.timeit(lambda: pandas.concat([pd_df, md_df]), number=10)
  print(f"\t{time}s.\n")
  # Possible output: TypeError


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
