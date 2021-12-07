:orphan:

IO Module Description
"""""""""""""""""""""

Dispatcher Classes Workflow Overview
''''''''''''''''''''''''''''''''''''

Call from ``read_*`` function of execution-specific IO class is forwarded to the
``_read`` function of file format-specific class, where function parameters are
preprocessed to check if they are supported (otherwise default pandas implementation
is used) and compute some metadata common for all partitions. Then file is splitted
into chunks (mechanism of splitting is described below) and using this data, tasks
are launched on the remote workers. After remote tasks are finished, additional
results postprocessing is performed, and new query compiler with imported data will
be returned.

Data File Splitting Mechanism
'''''''''''''''''''''''''''''

Modin file splitting mechanism differs depending on the data format type:

* text format type - file is splitted into bytes according user specified needs.
  In the simplest case, when no row related parameters (such as ``nrows`` or
  ``skiprows``) are passed, data chunks limits (start and end bytes) are derived
  by just roughly dividing the file size by the number of partitions (chunks can
  slightly differ between each other because usually end byte may occurs inside a
  line and in that case the last byte of the line should be used instead of initial
  value). In other cases the same splitting into bytes is used, but chunks sizes are
  defined according to the number of lines that each partition should contain.

* columnar store type - file is splitted by even distribution of columns that should
  be read between chunks.

* SQL type - chunking is obtained by wrapping initial SQL query into query that
  specifies initial row offset and number of rows in the chunk.

After file splitting is complete, chunks data is passed to the parser functions
(``PandasCSVParser.parse`` for ``read_csv`` function with pandas storage format) for
further processing on each worker.

Submodules Description
''''''''''''''''''''''

``modin.core.io`` module is used mostly for storing utils and dispatcher
classes for reading files of different formats.

* ``io.py`` - class containing basic utils and default implementation of IO functions.

* ``file_dispatcher.py`` - class reading data from different kinds of files and
  handling some util functions common for all formats. Also this class contains ``read``
  function which is entry point function for all dispatchers ``_read`` functions.

* text - directory for storing all text file format dispatcher classes  
  
  * ``text_file_dispatcher.py`` - class for reading text formats files. This class
    holds ``partitioned_file`` function for splitting text format files into chunks,
    ``offset`` function for moving file offset at the specified amount of bytes,
    ``_read_rows`` function for moving file offset at the specified amount of rows
    and many other functions.
  
  * format/feature specific dispatchers: ``csv_dispatcher.py``, ``csv_glob_dispatcher.py``
    (reading multiple files simultaneously, experimental feature), ``excel_dispatcher.py``,
    ``fwf_dispatcher.py`` and ``json_dispatcher.py``.

* column_stores - directory for storing all columnar store file format dispatcher classes
  
  * ``column_store_dispatcher.py`` - class for reading columnar type files. This class
    holds ``build_query_compiler`` function that performs file splitting, deploying remote
    tasks and results postprocessing and many other functions.
  
  * format/feature specific dispatchers: ``feather_dispatcher.py``, ``hdf_dispatcher.py``
    and ``parquet_dispatcher.py``.

* sql - directory for storing SQL dispatcher class
  
  * ``sql_dispatcher.py`` -  class for reading SQL queries or database tables.

Handling ``skiprows`` Parameter
'''''''''''''''''''''''''''''''

Handling ``skiprows`` parameter by pandas import functions can be very tricky, especially
for ``read_csv`` function because of interconnection with ``header`` parameter. In this section
the techniques of ``skiprows`` processing by both pandas and Modin are covered.

Processing ``skiprows`` by pandas
=================================

Let's consider a simple snippet with ``pandas.read_csv`` in order to understand interconnection
of ``header`` and ``skiprows`` parameters:

.. code-block:: python

  import pandas
  from io import StringIO

  data = """0
  1
  2
  3
  4
  5
  6
  7
  8
  """

  # `header` parameter absence is equivalent to `header="infer"` or `header=0`
  # rows 1, 5, 6, 7, 8 are read with header "0"
  df = pandas.read_csv(StringIO(data), skiprows=[2, 3, 4])
  # rows 5, 6, 7, 8 are read with header "1", row 0 is skipped additionally
  df = pandas.read_csv(StringIO(data), skiprows=[2, 3, 4], header=1)
  # rows 6, 7, 8 are read with header "5", rows 0, 1 are skipped additionally
  df = pandas.read_csv(StringIO(data), skiprows=[2, 3, 4], header=2)

In the examples above list-like ``skiprows`` values are fixed and ``header`` is varied. In the first
example with no ``header`` provided, rows 2, 3, 4 are skipped and row 0 is considered as a header.
In the second example ``header == 1``, so 0th row is skipped and the next available row is
considered as a header. The third example shows the case when ``header`` and ``skiprows`` parameters
values are intersected - in this case skipped rows are dropped first and only then ``header`` is got
from the remaining rows (rows before header are skipped too).

In the examples above only list-like ``skiprows`` and integer ``header`` parameters are considered,
but the same logic is applicable for other types of the parameters.

Processing ``skiprows`` by Modin
================================

As it can be seen, skipping rows in the pandas import functions is complicated and distributing
this logic across multiple workers can complicate it even more. Thus in some rare corner cases
default pandas implementation is used in Modin to avoid excessive Modin code complication.

Modin uses two techniques for skipping rows:

1) During file partitioning (setting file limits that should be read by each partition)
exact rows can be excluded from partitioning scope, thus they won't be read at all and can be
considered as skipped. This is the most effective way of skipping rows since it doesn't require
any actual data reading and postprocessing, but in this case ``skiprows`` parameter can be an
integer only. When it is possible Modin always uses this approach.

2) Rows for skipping can be dropped after full dataset import. This is more expensive way since
it requires extra IO work and postprocessing afterwards, but ``skiprows`` parameter can be of any
non-integer type supported by ``pandas.read_csv``.

In some cases, if ``skiprows`` is uniformly distributed array (e.g. [1, 2, 3]), ``skiprows`` can be
"squashed" and represented as an integer to make a fastpath by skipping these rows during file partitioning
(using the first option). But if there is a gap between the first row for skipping
and the last line of the header (that will be skipped too since header is read by each partition
to ensure metadata is defined properly), then this gap should be assigned for reading first
by assigning the first partition to read these rows by setting ``pre_reading`` parameter.

Let's consider an example of skipping rows during partitioning when ``header="infer"`` and
``skiprows=[3, 4, 5]``. In this specific case fastpath can be done since ``skiprows`` is uniformly
distributed array, so we can "squash" it to an integer and set "partitioning" skiprows to 3. But
if no additional action is done, these three rows will be skipped right after header line,
that corresponds to ``skiprows=[1, 2, 3]``. To avoid this discrepancy, we need to assign the first
partition to read data between header line and the first row for skipping by setting special
``pre_reading`` parameter to 2. Then, after the skipping of rows considered to be skipped during
partitioning, the rest data will be divided between the rest of partitions, see rows assignment
below:

.. code-block::

  0 - header line (skip during partitioning)
  1 - pre reading (assign to read by the first partition)
  2 - pre reading (assign to read by the first partition)
  3 - "partitioning" skiprows (skip during partitioning)
  4 - "partitioning" skiprows (skip during partitioning)
  5 - "partitioning" skiprows (skip during partitioning)
  6 - data to partition (divide between the rest of partitions)
  7 - data to partition (divide between the rest of partitions)