Optimization Notes
==================

Modin has chosen default values for a lot of the configurations here that provide excellent performance in most
cases. This page is for those who love to optimize their code and those who are curious about existing optimizations
within Modin. Here you can find more information about Modin's optimizations both for a pipeline of operations as
well as for specific operations.

Understanding Modin's partitioning mechanism
""""""""""""""""""""""""""""""""""""""""""""

Modin's partitioning is crucial for performance; so we recommend expert users to understand Modin's
partitioning mechanism and how to tune it in order to achieve better performance.

How Modin partitions a dataframe
--------------------------------

Modin uses a partitioning scheme that partitions a dataframe along both axes, resulting in a matrix
of partitions. The row and column chunk sizes are computed independently based
on the length of the appropriate axis and Modin's special :doc:`configuration variables </flow/modin/config>`
(``NPartitions`` and ``MinPartitionSize``):

- ``NPartitions`` is the maximum number of splits along an axis; by default, it equals to the number of cores
  on your local machine or cluster of nodes.
- ``MinPartitionSize`` is the minimum number of rows/columns to do a split. For instance, if ``MinPartitionSize``
  is 32, the column axis will not be split unless the amount of columns is greater than 32. If it is is greater, for example, 34,
  then the column axis is sliced into two partitions: containing 32 and 2 columns accordingly.

Beware that ``NPartitions`` specifies a limit for the number of partitions `along a single axis`, which means, that
the actual limit for the entire dataframe itself is the square of ``NPartitions``.

.. figure:: /img/partitioning_mechanism/partitioning_examples.svg
   :align: center

Full-axis functions
-------------------

Some of the aggregation functions require knowledge about the entire axis, for example at ``.apply(foo, axis=0)``
the passed function ``foo`` expects to receive data for the whole column at once.

When a full-axis function is applied, the partitions along this axis are collected at a single worker
that processes the function. After the function is done, the partitioning of the data is back to normal.

.. figure:: /img/partitioning_mechanism/full_axis_function.svg
   :align: center

Note that the amount of remote calls is equal to the number of partitions, which means that since the number
of partitions is decreased for full-axis functions it also decreases the potential for parallelism.

Also note, that reduce functions such as ``.sum()``, ``.mean()``, ``.max()``, etc, are not considered
to be full-axis, so they do not suffer from the decreasing level of parallelism.

How to tune partitioning
------------------------

As you can see from the examples above, the more the dataframe's shape is closer to a square, the closer the number of
partitions to the square of ``NPartitions``. In the case of ``NPartitions`` equals to the number of workers,
that means that a single worker is going to process multiple partitions at once, which slows down overall performance.

If your workflow mainly operates with wide dataframes and non-full-axis functions, it makes sense to reduce the
``NPartitions`` value so a single worker would process a single partition.

.. figure:: /img/partitioning_mechanism/repartition_square_frames.svg
   :align: center

Copy-pastable example, showing how tuning ``NPartitions`` value for wide frames may improve performance on your machine:

.. code-block:: python

  from multiprocessing import cpu_count
  from modin.distributed.dataframe.pandas import unwrap_partitions
  import modin.config as cfg
  import modin.pandas as pd
  import numpy as np
  import timeit

  # Generating data for a square-like dataframe
  data = np.random.randint(0, 100, size=(5000, 5000))

  # Explicitly setting `NPartitions` to its default value
  cfg.NPartitions.put(cpu_count())

  # Each worker processes `cpu_count()` amount of partitions
  df = pd.DataFrame(data)
  print(f"NPartitions: {cfg.NPartitions.get()}")
  # Getting raw partitions to count them
  partitions_shape = np.array(unwrap_partitions(df)).shape
  print(
      f"The frame has {partitions_shape[0]}x{partitions_shape[1]}={np.prod(partitions_shape)} partitions "
      f"when the CPU has only {cpu_count()} cores."
  )
  print(f"10 times of .abs(): {timeit.timeit(lambda: df.abs(), number=10)}s.")
  # Possible output:
  #   NPartitions: 112
  #   The frame has 112x112=12544 partitions when the CPU has only 112 cores.
  #   10 times of .abs(): 23.64s.

  # Taking a square root of the the current `cpu_count` to make more even partitioning
  cfg.NPartitions.put(int(cpu_count() ** 0.5))

  # Each worker processes a single partition
  df = pd.DataFrame(data)
  print(f"NPartitions: {cfg.NPartitions.get()}")
  # Getting raw partitions to count them
  partitions_shape = np.array(unwrap_partitions(df)).shape
  print(
      f"The frame has {partitions_shape[0]}x{partitions_shape[1]}={np.prod(partitions_shape)} "
      f"when the CPU has {cpu_count()} cores."
  )
  print(f"10 times of .abs(): {timeit.timeit(lambda: df.abs(), number=10)}s.")
  # Possible output:
  #   NPartitions: 10
  #   The frame has 10x10=100 partitions when the CPU has 112 cores.
  #   10 times of .abs(): 0.25s.


Avoid iterating over Modin DataFrame
""""""""""""""""""""""""""""""""""""

Use ``df.apply()`` or other aggregation methods when possible instead of iterating over a dataframe.
For-loops don't scale and forces the distributed data to be collected back at the driver.

Copy-pastable example, showing how replacing a for-loop to the equivalent ``.apply()`` may improve performance:

.. code-block:: python

  import modin.pandas as pd
  import numpy as np
  from timeit import default_timer as timer

  data = np.random.randint(1, 100, (2 ** 10, 2 ** 2))

  md_df = pd.DataFrame(data)

  result = []
  t1 = timer()
  # Iterating over a dataframe forces to collect distributed data to the driver and doesn't scale
  for idx, row in md_df.iterrows():
      result.append((row[1] + row[2]) / row[3])
  print(f"Filling a list by iterating a Modin frame: {timer() - t1:.2f}s.")
  # Possible output: 36.15s.

  t1 = timer()
  # Using `.apply()` perfectly scales to all axis-partitions
  result = md_df.apply(lambda row: (row[1] + row[2]) / row[3], axis=1).to_numpy().tolist()
  print(f"Filling a list by using '.apply()' and converting the result to a list: {timer() - t1:.2f}s.")
  # Possible output: 0.22s.

Use Modin's Dataframe Algebra API to implement custom parallel functions
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

Modin provides a set of low-level parallel-implemented operators which can be used to build most of the
aggregation functions. These operators are present in the :doc:`algebra module </flow/modin/core/dataframe/algebra>`.
Modin DataFrame allows users to use their own aggregations built with this module. Visit the
:doc:`appropriate section </flow/modin/core/dataframe/algebra>` of the documentation for the steps to do it.

Avoid mixing pandas and Modin DataFrames
""""""""""""""""""""""""""""""""""""""""

Although Modin is considered to be a drop-in replacement for pandas, Modin and pandas are not intended to be used together
in a single flow. Passing a pandas DataFrame as an argument for a Modin's DataFrame method may either slowdown
the function (because it has to process non-distributed object) or raise an error. You would also get an undefined
behavior if you pass a Modin DataFrame as an input to pandas methods, since pandas identifies Modin's objects as a simple iterable,
and so can't leverage its benefits as a distributed dataframe.

Copy-pastable example, showing how mixing pandas and Modin DataFrames in a single flow may bottleneck performance:

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


Operation-specific optimizations
""""""""""""""""""""""""""""""""

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
