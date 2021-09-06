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
  from modin.config import NPartitions

  NPartitions.put(2)

  import modin.pandas as pd

  test_filename = "test.csv"

  # data with empty (blank) line
  data = """col1,col2
  1,2
  3,4
  5,6

  7,8
  """
  kwargs = {
      "header": None, # forces to not infer header from file
      "skip_blank_lines": False, # forces to handle blank lines as NaNs
      # "dtype": "str", # workaround to force partitions to read data with correct types 
  }


  try:
      with open(test_filename, "w") as f:
          f.write(data)
      
      pandas_df = pandas.read_csv(test_filename, **kwargs)
      pd_df = pd.read_csv(test_filename, **kwargs)

      print(pandas_df.iloc[1, 0], type(pandas_df.iloc[1, 0])) # 1 <class 'str'>
      print(pandas_df.iloc[4, 0], type(pandas_df.iloc[4, 0])) # 5 <class 'str'>
      print(pd_df.iloc[1, 0], type(pd_df.iloc[1, 0])) # 1st partition: 1 <class 'str'>
      print(pd_df.iloc[4, 0], type(pd_df.iloc[4, 0])) # 2nd partition: 5.0 <class 'numpy.float64'>
  finally:
    os.remove(test_filename)

In this case `DataFrame` read by pandas ``pandas_df`` contain only `str` data (integers were casted
to strings because `header = None` that forced to read header line in the data fields and handle all the
data as strings). Modin the fisrt partition read data similary to pandas, but the second partition doesn't
read header line (that doesn't forced to cast data to strings), so data was casted to float type (integers
were casted to float type because of blank lines which are handled as float type NaNs) which, in turn,
forces `read_csv` to add `.0` symbols to integers in the second partition. Resulting pandas and Modin frames
will be next:

 But in the of Modin case, 

.. code-block:: python

  pandas_df:
          0     1
    0  col1  col2
    1     1     2
    2     3     4
    3     5     6
    4   NaN   NaN
    5     7     8

    pd_df:
          0     1
    0  col1  col2
    1     1     2
    2     3     4
    3   5.0   6.0
    4   NaN   NaN
    5   7.0   8.0

The above example showed the mechanism of occurence of pandas and Modin ``read_csv`` outputs
discrepancy during heterogeneous data import. Please note, that similar situations can occur
during different data/parameters combinations.

**Solution**

In the case if heterogeneous data is detected, corresponding warning will be showed to the
user. Unfortunetely, the fact, that data is heterogeneous, can be detected only after full data
set is already imported and subsequent data types casting brings significant performance
degradation, so, by default types casting is disabled in Modin. If it is needed, it can be
enabled by setting environment variable `MODIN_DO_TYPES_CAST` (or Modin config
`DoTypesCastOnImport`) to `True` value.

In order to avoid such performance degradation and still obtain matched partitions data types,
it is prefferable to set `dtype` parameter of `read_csv` - this will force partitions to read
data with correct type and avoid excessive types casting afterward. To reduce performance
degradation try to set `dtype` value as fine-grained as it possible (specify `dtype` parameter
only for columns with heterogeneous data).

Setting of `dtype` parameter works well for most of the cases, but, unfortunetely, it is
ineffective if data file contain column which should be interpreted as index (`index_col`
parameter is used) since `dtype` parameter is responsible only for data fields. In this case
data set should be imported without setting of `index_col` parameter and only then index
column should be set as index (by using `DataFrame.set_index` funcion for example) as it
shown in the example below:

.. code-block:: python

  pd_df = pd.read_csv(filename, dtype=data_dtype, index_col=None)
  pd_df = df_df.set_index(index_col_name)

.. _issue: https://github.com/modin-project/modin/issues
