Troubleshooting
===============

We hope your experience with Modin is bug-free, but there are some quirks about Modin
that may require troubleshooting. If you are still having issues, please post on
the #support channel on our Slack_ community or open a Github issue_.

Frequently encountered issues
-----------------------------

This is a list of the most frequently encountered issues when using Modin. Some of these
are working as intended, while others are known bugs that are being actively worked on.

Warning during execution: ``defaulting to pandas``
""""""""""""""""""""""""""""""""""""""""""""""""""

Please note, that while Modin covers a large portion of the pandas API, not all functionality is implemented. For methods that are not yet implemented, such as ``asfreq``, you may see the following:

.. code-block:: text

  UserWarning: `DataFrame.asfreq` defaulting to pandas implementation.

To understand which functions will lead to this warning, we have compiled a list of :doc:`currently supported methods </supported_apis/index>`. When you see this warning, Modin defaults to pandas by converting the Modin dataframe to pandas to perform the operation. Once the operation is complete in pandas, it is converted back to a Modin dataframe. These operations will have a high overhead due to the communication involved and will take longer than pandas. When this is happening, a warning will be given to the user to inform them that this operation will take longer than usual. You can learn more about this :doc:`here </supported_apis/defaulting_to_pandas>`.

If you would like to request a particular method be implemented, feel free to open an
`issue`_. Before you open an issue please make sure that someone else has not already
requested that functionality.

Hanging on ``import modin.pandas as pd``
""""""""""""""""""""""""""""""""""""""""

This can happen when Ray fails to start. It will keep retrying, but often it is faster
to just restart the notebook or interpreter. Generally, this should not happen. Most
commonly this is encountered when starting multiple notebooks or interpreters in quick
succession.

**Solution**

Restart your interpreter or notebook kernel.

**Avoiding this Error**

Avoid starting many Modin notebooks or interpreters in quick succession. Wait 2-3
seconds before starting the next one.

Importing heterogeneous data using ``read_csv``
"""""""""""""""""""""""""""""""""""""""""""""""

Since Modin's ``read_csv`` imports data in parallel, it is possible for data across
partitions to be heterogeneously typed (this can happen when columns contain
heterogeneous data, i.e. values in the same column are of different types). An example
of how this is handled is shown below.

.. code-block:: python

  import os
  import pandas
  import modin.pandas as pd
  from modin.config import NPartitions

  NPartitions.put(2)

  test_filename = "test.csv"
  # data with heterogeneous values in the first column
  data = """one,2
  3,4
  5,6
  7,8
  9.0,10
  """
  kwargs = {
      # names of the columns to set, if `names` parameter is set,
      # header inffering from the first data row/rows will be disabled
      "names": ["col1", "col2"],

      # explicit setting of data type of column/columns with heterogeneous
      # data will force partitions to read data with correct dtype
      # "dtype": {"col1": str},
  }


  try :
      with open(test_filename, "w") as f:
          f.write(data)

      pandas_df = pandas.read_csv(test_filename, **kwargs)
      pd_df = pd.read_csv(test_filename, **kwargs)

      print(pandas_df)
      print(pd_df)
  finally:
      os.remove(test_filename)

  Output:

  pandas_df:
    col1  col2
  0  one     2
  1    3     4
  2    5     6
  3    7     8
  4  9.0    10

  pd_df:
    col1  col2
  0  one     2
  1    3     4
  2    5     6
  3  7.0     8
  4  9.0    10


In this case, ``col1`` of the `DataFrame` read by pandas contains only ``str`` data
because the first value ("one") is inferred to have type ``str``, which forces pandas to handle the rest of the values in the column
as strings. The first Modin partition (the first three rows) handles the data as pandas does,
but the second partition (the last two rows) reads the data as floats. This is because the
second column contains an int and a float, and thus the column type is inferred to be float. As a
result, `7` is interpreted as `7.0`, which differs from the pandas output.

The above example demonstrates heterogenous data import with str, int, and float types,
but heterogeneous data consisting of other data/parameter combinations can also result in 
data type mismatches with pandas.

**Solution**

When heterogeneous data is detected, a warning will be raised.
Currently, these discrepancies aren't properly handled
by Modin, so to avoid this issue, you need to set the ``dtype`` parameter of ``read_csv``
manually to force the correct data type coercion during data import. Note that 
to avoid excessive performance degradation, the ``dtype`` value should only be set for columns that may contain heterogenous data.
as possible (specify ``dtype`` parameter only for columns with heterogeneous data).

Specifying the ``dtype`` parameter will work well in most cases. If the file
contains a column that should be interpreted as the index
(the ``index_col`` parameter is specified) there may still be type discrepancies in the index, since the ``dtype`` parameter is only responsible for data
fields. If in the above example, ``kwargs`` was set like so:

.. code-block:: python

  kwargs = {
      "names": ["col1", "col2"],
      "dtype": {"col1": str},
      "index_col": "col1",
  }

The resulting Modin DataFrame will contain incorrect values - just as if ``dtype``
had not been specified:

.. code-block:: python

  col1
  one      2
  3        4
  5        6
  7.0      8
  9.0     10

One workaround is to import the data without setting the ``index_col`` parameter, and then 
set the index column using the ``DataFrame.set_index`` function as shown in
the example below:

.. code-block:: python

  pd_df = pd.read_csv(filename, dtype=data_dtype, index_col=None)
  pd_df = pd_df.set_index(index_col_name)
  pd_df.index.name = None


Using Modin with python multiprocessing
"""""""""""""""""""""""""""""""""""""""

We strongly recommend against using a distributed execution engine (e.g. Ray or Dask)
in conjunction with Python multiprocessing because that can lead to undefined behavior.
One such example is shown below:

.. code-block:: python

  import modin.pandas as pd

  # Ray engine is used by default
  df = pandas.DataFrame([1, 2, 3])

  def f(arg):
    return df + arg

  if __name__ == '__main__':
    from multiprocessing import Pool

    with Pool(5) as p:
        print(p.map(f, [1]))

Although this example may work on your machine, we do not recommend it, because
the Python multiprocessing library will duplicate Ray clusters, causing both
excessive resource usage and conflict over the available resources.

Poor performance of the first operation with Modin on Ray engine
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

There might be cases when the first operation with Modin on Ray engine is much slower than the subsequent calls of the operation.
That happens because Ray workers may not be fully set up yet to perform computation after initialization of the engine
with ``ray.init(runtime_env={'env_vars': {'__MODIN_AUTOIMPORT_PANDAS__': '1'}})``, which is the default behavior of Modin on Ray engine
if Ray has not been initialised yet. Modin intentionaly initializes Ray this way to import ``pandas`` in workers
once Python interpreter is started in them so that to avoid a race condition in Ray between the import thread and the thread executing the code.

..
      See more details on why we started using ``ray.init(runtime_env={'env_vars': {'__MODIN_AUTOIMPORT_PANDAS__': '1'}})` in
      https://github.com/modin-project/modin/pull/4603.

.. code-block:: python

  import time
  import pandas
  import numpy as np
  import ray
  import modin.pandas as pd
  import modin.config as cfg

  # Look at the Ray documentation with respect to the Ray configuration suited to you most.
  ray.init(runtime_env={'env_vars': {'__MODIN_AUTOIMPORT_PANDAS__': '1'}})

  pandas_df = pandas.DataFrame(
    np.random.randint(0, 100, size=(1000000, 13))
  )
  pandas_df.to_csv("foo.csv", index=False)

  def read_csv_with_pandas():
    start_time = time.time()
    pandas_df = pandas.read_csv("foo.csv", index_col=0)
    end_time = time.time()
    pandas_duration = end_time - start_time
    print("Time to read_csv with pandas: {} seconds".format(round(pandas_duration, 3)))
    return pandas_df

  def read_csv_with_modin():
    start_time = time.time()
    modin_df = pd.read_csv("foo.csv", index_col=0)
    end_time = time.time()
    modin_duration = end_time - start_time
    print("Time to read_csv with Modin: {} seconds".format(round(modin_duration, 3))) 
    return modin_df

  for i in range(5):
    pandas_df = read_csv_with_pandas()
    modin_df = read_csv_with_modin()

  Time to read_csv with pandas: 0.708 seconds
  Time to read_csv with Modin: 4.132 seconds
  Time to read_csv with pandas: 0.735 seconds
  Time to read_csv with Modin: 0.37 seconds
  Time to read_csv with pandas: 0.646 seconds
  Time to read_csv with Modin: 0.377 seconds
  Time to read_csv with pandas: 0.673 seconds
  Time to read_csv with Modin: 0.371 seconds
  Time to read_csv with pandas: 0.672 seconds
  Time to read_csv with Modin: 0.379 seconds

**Solution**

So far there is no a solution to fix or work around the problem rather than not to pass a non-empty runtime_env to ``ray.init()``.
However, this may lead to other problem regarding a race condition in Ray between the import thread and the thread executing the code.
So for now we just highlight the problem in hope of a future fix in Ray itself.

Also, it is worth noting that every distributed engine by its nature has a little overhead for the first operation being called,
which may be important for microbenchmarks. What you likely want to do is warm up worker processes
either by excluding the time of the first iteration from your measurements or execute a simple function in workers to fully set up them.

Common errors
-------------

Error when using Dask engine: ``RuntimeError: if __name__ == '__main__':``
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

The following `script.py` uses Modin with Dask as an execution engine and produces errors:

.. code-block:: python

  # script.py
  import modin.pandas as pd
  import modin.config as cfg

  cfg.Engine.put("dask")

  df = pd.DataFrame([0,1,2,3])
  print(df)

A part of the produced errors by the script above would be the following:

.. code-block::

  File "/path/python3.9/multiprocessing/spawn.py", line 134, in _check_not_importing_main
    raise RuntimeError('''
    RuntimeError: 
        An attempt has been made to start a new process before the
        current process has finished its bootstrapping phase.

        This probably means that you are not using fork to start your
        child processes and you have forgotten to use the proper idiom
        in the main module:

            if __name__ == '__main__':
                freeze_support()
                ...

        The "freeze_support()" line can be omitted if the program
        is not going to be frozen to produce an executable.

This happens because Dask Client uses `fork <https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods>`_
to start processes.

**Solution**

To avoid the problem the Dask Client creation code needs to be moved into the ``__main__`` scope of the module.

The corrected `script.py` would look like:

.. code-block:: python

  # script.py
  import modin.pandas as pd
  import modin.config as cfg

  cfg.Engine.put("dask")

  if __name__ == "__main__":
    df = pd.DataFrame([0, 1, 2, 3]) # Dask Client creation is hidden in the first call of Modin functionality.
    print(df)

or

.. code-block:: python

  # script.py
  from distributed import Client
  import modin.pandas as pd
  import modin.config as cfg

  cfg.Engine.put("dask")

  if __name__ == "__main__":
    # Explicit Dask Client creation.
    # Look at the Dask Distributed documentation with respect to the Client configuration suited to you most.
    client = Client()
    df = pd.DataFrame([0, 1, 2, 3])
    print(df)

Spurious error "cannot import partially initialised pandas module" on custom Ray cluster
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

If you're using some pre-configured Ray cluster to run Modin, it's possible you would
be seeing spurious errors like

.. code-block::

  ray.exceptions.RaySystemError: System error: partially initialized module 'pandas' has no attribute 'core' (most likely due to a circular import)
  traceback: Traceback (most recent call last):
    File "/usr/share/miniconda/envs/modin/lib/python3.8/site-packages/ray/serialization.py", line 340, in deserialize_objects
      obj = self._deserialize_object(data, metadata, object_ref)
    File "/usr/share/miniconda/envs/modin/lib/python3.8/site-packages/ray/serialization.py", line 237, in _deserialize_object
      return self._deserialize_msgpack_data(data, metadata_fields)
    File "/usr/share/miniconda/envs/modin/lib/python3.8/site-packages/ray/serialization.py", line 192, in _deserialize_msgpack_data
      python_objects = self._deserialize_pickle5_data(pickle5_data)
    File "/usr/share/miniconda/envs/modin/lib/python3.8/site-packages/ray/serialization.py", line 180, in _deserialize_pickle5_data
      obj = pickle.loads(in_band, buffers=buffers)
    File "/usr/share/miniconda/envs/modin/lib/python3.8/site-packages/pandas/__init__.py", line 135, in <module>
      from pandas import api, arrays, errors, io, plotting, testing, tseries
    File "/usr/share/miniconda/envs/modin/lib/python3.8/site-packages/pandas/testing.py", line 6, in <module>
      from pandas._testing import (
    File "/usr/share/miniconda/envs/modin/lib/python3.8/site-packages/pandas/_testing/__init__.py", line 979, in <module>
      cython_table = pd.core.common._cython_table.items()
  AttributeError: partially initialized module 'pandas' has no attribute 'core' (most likely due to a circular import)

**Solution**

Modin contains a workaround that should automatically do ``import pandas`` upon worker process starts.

It is triggered by the presence of non-empty ``__MODIN_AUTOIMPORT_PANDAS__`` environment variable which
Modin sets up automatically on the Ray clusters it spawns, but it might be missing on pre-configured clusters.

So if you're seeing the issue like shown above, please make sure you set this environment variable on all
worker nodes of your cluster before actually spawning the workers.

.. _issue: https://github.com/modin-project/modin/issues
.. _Slack: https://modin.org/slack.html
