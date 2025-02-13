Optimization Notes
==================

Modin has chosen default values for a lot of the configurations here that provide excellent performance in most
cases. This page is for those who love to optimize their code and those who are curious about existing optimizations
within Modin. Here you can find more information about Modin's optimizations both for a pipeline of operations as
well as for specific operations. If you want to go ahead and tune the Modin behavior on your own, refer to
:doc:`Modin Configuration Settings </flow/modin/config>` page for the full set of configurations available in Modin.

Range-partitioning in Modin
"""""""""""""""""""""""""""

Modin utilizes a range-partitioning approach for specific operations, significantly enhancing
parallelism and reducing memory consumption in certain scenarios. Range-partitioning is typically
engaged for operations that has key columns (to group on, to merge on, etc).

You can enable `range-partitioning`_ by specifying ``cfg.RangePartitioning`` :doc:`configuration variable: </flow/modin/config>`

.. code-block:: python

    import modin.pandas as pd
    import modin.config as cfg

    cfg.RangePartitioning.put(True) # past this point methods that support range-partitioning
                                    # will use it

    pd.DataFrame(...).groupby(...).mean() # use range-partitioning for groupby.mean()

    cfg.Range-partitioning.put(False)

    pd.DataFrame(...).groupby(...).mean() # use MapReduce implementation for groupby.mean()

Building range-partitioning assumes data reshuffling, which may result into breaking the original
order of rows, for some operation, it will mean that the result will be different from Pandas.

Range-partitioning is not a silver bullet, meaning that enabling it is not always beneficial. Below you find
a link to the list of operations that have support for range-partitioning and practical advices on when one should
enable it: :doc:`operations that support range-partitioning </usage_guide/optimization_notes/range_partitioning_ops>`.

Dynamic-partitioning in Modin
"""""""""""""""""""""""""""""

Ray engine experiences slowdowns when running a large number of small remote tasks at the same time. Ray Core recommends to `avoid tiny task`_.
When modin DataFrame has a large number of partitions, some functions produce a large number of remote tasks, which can cause slowdowns. 
To solve this problem, Modin suggests using dynamic partitioning. This approach reduces the number of remote tasks 
by combining multiple partitions into a single virtual partition and perform a common remote task on them.

Dynamic partitioning is typically used for operations that are fully or partially executed on all partitions separately.

.. code-block:: python

    import modin.pandas as pd
    from modin.config import context

    df = pd.DataFrame(...)

    with context(DynamicPartitioning=True):
        df.abs()

Dynamic partitioning is also not always useful, and this approach is usually used for medium-sized DataFrames with a large number of columns.
If the number of columns is small, the number of partitions will be close to the number of CPUs, and Ray will not have this problem.
If the DataFrame has too many rows, this is also not a good case for using Dynamic-partitioning, since each task is no longer tiny and performing 
the combined tasks carries more overhead than assigning them separately.

Unfortunately, the use of Dynamic-partitioning depends on various factors such as data size, number of CPUs, operations performed, 
and it is up to the user to determine whether Dynamic-partitioning will give a boost in his case or not.

..
  TODO: Define heuristics to automatically enable dynamic partitioning without performance penalty.
  `Issue #7370 <https://github.com/modin-project/modin/issues/7370>`_

Understanding Modin's partitioning mechanism
""""""""""""""""""""""""""""""""""""""""""""

Modin's partitioning is crucial for performance; so we recommend expert users to understand Modin's
partitioning mechanism and how to tune it in order to achieve better performance.

How Modin partitions a dataframe
--------------------------------

Modin uses a partitioning scheme that partitions a dataframe along both axes, resulting in a matrix
of partitions. The row and column chunk sizes are computed independently based
on the length of the appropriate axis and Modin's special :doc:`configuration variables </flow/modin/config>`
(``NPartitions``, ``MinRowPartitionSize`` and ``MinColumnPartitionSize``):

- ``NPartitions`` is the maximum number of splits along an axis; by default, it equals to the number of cores
  on your local machine or cluster of nodes.
- ``MinRowPartitionSize`` is the minimum number of rows to do a split. For instance, if ``MinRowPartitionSize``
  is 32, the row axis will not be split unless the amount of rows is greater than 32. If it is is greater, for example, 34,
  then the row axis is sliced into two partitions: containing 32 and 2 rows accordingly.
- ``MinColumnPartitionSize`` is the minimum number of columns to do a split. For instance, if ``MinColumnPartitionSize``
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

Configure Modin's default partitioning scheme
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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

Manually trigger repartitioning
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you're getting unexpectedly poor performance, although you configured ``MODIN_NPARTITIONS``
correctly, then this might be caused by unbalanced partitioning that occurred during the
workflow's execution.

Modin's idealogy is to handle partitioning internally and not let users worry about the possible
consequences of applying a lot of "bad" operations that may affect DataFrame's partitioning.
We're constantly making efforts to find and fix cases where partitioning may cause a headache
for users.

However, if you feel that you're dealing with unbalanced partitioning you may try to call an
internal :py:meth:`modin.pandas.dataframe.DataFrame._repartition` method on your :py:class:`~modin.pandas.dataframe.DataFrame` in order to manually
trigger partitions rebalancing and see whether it improves performance for your case.

.. automethod:: modin.pandas.dataframe.DataFrame._repartition

An actual use-case for this method may be the following:

.. code-block:: python

  import modin.pandas as pd
  import timeit

  df = pd.DataFrame({"col0": [1, 2, 3, 4]})

  # Appending a lot of columns may result into unbalanced partitioning
  for i in range(1, 128):
      df[f"col{i}"] = pd.Series([1, 2, 3, 4])

  print(
      "DataFrame with unbalanced partitioning:",
      timeit.timeit(lambda: df.sum(), number=10)
  ) # 1.44s

  df = df._repartition()
  print(
      "DataFrame after '._repartition()':",
      timeit.timeit(lambda: df.sum(), number=10)
  ) # 0.21s.

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
:doc:`DataFrame's algebra </flow/modin/core/dataframe/algebra>` page of the documentation for the steps to do it.

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


Using pandas to execute queries in Modin's ``"native"`` execution mode
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

By default, Modin distributes the data in a dataframe (or series) and attempts
to process data for different partitions in parallel.

However, for certain scenarios, such as handling small datasets, Modin's
parallel execution may introduce unnecessary overhead. In such cases, it's more
efficient to use serial execution with a single, unpartitioned pandas dataframe.
You can enable this kind of "native" execution by setting Modin's
``StorageFormat`` and ``Engine``
:doc:`configuration variables </flow/modin/config>` to ``"Native"``.

DataFrames created while Modin's global execution mode is set to ``"Native"``
will continue to use native execution even if you switch the execution mode
later. Modin supports interoperability between distributed Modin DataFrames
and those using native execution.

Here is an example of using native execution:

.. code-block:: python

  import modin.pandas as pd
  from modin import set_execution
  from modin.config import StorageFormat, Engine

  # This dataframe will use Modin's default, distributed execution.
  df_distributed_1 = pd.DataFrame([0])
  assert df_distributed_1._query_compiler.engine != "Native"

  # Set execution to "Native" for native execution.
  original_engine, original_storage_format = set_execution(
    engine="Native",
    storage_format="Native"
  )
  native_df = pd.DataFrame([1])
  assert native_df._query_compiler.engine == "Native"

  # Revert to default settings for distributed execution
  set_execution(engine=original_engine, storage_format=original_storage_format)
  df_distributed_2 = pd.DataFrame([2])
  assert df_distributed_2._query_compiler.engine == original_engine

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

.. _range-partitioning: https://www.techopedia.com/definition/31994/range-partitioning
.. _`avoid tiny task`: https://docs.ray.io/en/latest/ray-core/tips-for-first-time.html#tip-2-avoid-tiny-tasks
