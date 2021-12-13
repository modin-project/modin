Troubleshooting
===============

We hope your experience with Modin is bug-free, but there are some quirks about Modin
that may require troubleshooting.

Frequently encountered issues
-----------------------------

This is a list of the most frequently encountered issues when using Modin. Some of these
are working as intended, while others are known bugs that are being actively worked on.

Error During execution: ``ArrowIOError: Broken Pipe``
"""""""""""""""""""""""""""""""""""""""""""""""""""""

One of the more frequently encountered issues is an ``ArrowIOError: Broken Pipe``. This
error can happen in a couple of different ways. One of the most common ways this is
encountered is from pressing **CTRL + C** sending a ``KeyboardInterrupt`` to Modin. In
Ray, when a ``KeyboardInterrupt`` is sent, Ray will shutdown. This causes the
``ArrowIOError: Broken Pipe`` because there is no longer an available plasma store for
working on remote tasks. This is working as intended, as it is not yet possible in Ray
to kill a task that has already started computation.

The other common way this ``Error`` is encountered is to let your computer go to sleep.
As an optimization, Ray will shutdown whenever the computer goes to sleep. This will
result in the same issue as above, because there is no longer a running instance of the
plasma store.

**Solution**

Restart your interpreter or notebook kernel.

**Avoiding this Error**

Avoid using ``KeyboardInterrupt`` and keeping your notebook or terminal running while
your machine is asleep. If you do ``KeyboardInterrupt``, you must restart the kernel or
interpreter.

Error during execution: ``ArrowInvalid: Maximum size exceeded (2GB)``
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

Encountering this issue means that the limits of the Arrow plasma store have been
exceeded by the partitions of your data. This can be encountered during shuffling data
or operations that require multiple datasets. This will only affect extremely large
DataFrames, and can potentially be worked around by setting the number of partitions.
This error is being actively worked on and should be resolved in a future release.

**Solution**

.. code-block:: python

  import modin.pandas as pd
  pd.DEFAULT_NPARTITIONS = 2 * pd.DEFAULT_NPARTITIONS

This will set the number of partitions to a higher count, and reduce the size in each.
If this does not work for you, please open an issue_.

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

Importing heterogeneous data by ``read_csv``
""""""""""""""""""""""""""""""""""""""""""""

Since Modin ``read_csv`` imports data in parallel, it can occur that data read by
different partitions can have different type (this happens when columns contains
heterogeneous data, i.e. column values are of different types), which are handled
differntly. Example of such behaviour is shown below.

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


In this case `DataFrame` read by pandas in the column ``col1`` contain only ``str`` data
because of the first string value ("one"), that forced pandas to handle full column
data as strings. Modin the fisrt partition (the first three rows) read data similary
to pandas, but the second partition (the last two rows) doesn't contain any strings
in the first column and it's data is read as floats because of the last column
value and as a result `7` value was read as `7.0`, that differs from pandas output.

The above example showed the mechanism of occurence of pandas and Modin ``read_csv``
outputs discrepancy during heterogeneous data import. Please note, that similar
situations can occur during different data/parameters combinations.

**Solution**

In the case if heterogeneous data is detected, corresponding warning will be showed in
the user's console. Currently, the discrepancies of such type doesn't properly handled
by Modin, and to avoid this issue, it is needed to set ``dtype`` parameter of ``read_csv``
function manually to force correct data type definition during data import by
partitions. Note, that to avoid excessive performance degradation, ``dtype`` value should
be set fine-grained as it possible (specify ``dtype`` parameter only for columns with
heterogeneous data).

Setting of ``dtype`` parameter works well for most of the cases, but, unfortunetely, it is
ineffective if data file contain column which should be interpreted as index
(``index_col`` parameter is used) since ``dtype`` parameter is responsible only for data
fields. For example, if in the above example, ``kwargs`` will be set in the next way:

.. code-block:: python

  kwargs = {
      "names": ["col1", "col2"],
      "dtype": {"col1": str},
      "index_col": "col1",
  }

Resulting Modin DataFrame will contain incorrect value as in the case if ``dtype``
is not set:

.. code-block:: python

  col1
  one      2
  3        4
  5        6
  7.0      8
  9.0     10

In this case data should be imported without setting of ``index_col`` parameter
and only then index column should be set as index (by using ``DataFrame.set_index``
funcion for example) as it is shown in the example below:

.. code-block:: python

  pd_df = pd.read_csv(filename, dtype=data_dtype, index_col=None)
  pd_df = pd_df.set_index(index_col_name)
  pd_df.index.name = None

Error when using OmniSci engine along with ``pyarrow.gandiva``: ``LLVM ERROR: inconsistency in registered CommandLine options``
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

This can happen when you use OmniSci engine along with ``pyarrow.gandiva``:

.. code-block:: python

  import modin.config as cfg
  cfg.Engine.put("Native")  # 'omniscidbe'/'dbe' would be imported with dlopen flags
  cfg.StorageFormat.put("Omnisci")
  cfg.IsExperimental.put(True)
  import modin.pandas as pd
  import pyarrow.gandiva as gandiva  # Error
  CommandLine Error: Option 'enable-vfe' registered more than once!
  LLVM ERROR: inconsistency in registered CommandLine options
  Aborted (core dumped)

**Solution**

Do not use OmniSci engine along with ``pyarrow.gandiva``.

.. _issue: https://github.com/modin-project/modin/issues

