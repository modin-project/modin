:orphan:

Pandas Parsers Module Description
"""""""""""""""""""""""""""""""""
High-Level Module Overview
''''''''''''''''''''''''''

This module houses parser classes (classes that are used for data parsing on the workers)
and util functions for handling parsing results. ``PandasParser`` is base class for parser
classes with pandas backend, that contains methods common for all child classes. Other
module classes implement ``parse`` function that performs parsing of specific format data
basing on the chunk information computed in the ``modin.engines.base.io`` module. After
chunk data parsing is completed, resulting ``DataFrame``-s will be splitted into smaller
``DataFrame``-s according to ``num_splits`` parameter, data type and number or
rows/columns in the parsed chunk, and then these frames and some additional metadata will
be returned.

Data Parsing Mechanism
''''''''''''''''''''''

Data parsing mechanism differs depending on the data format type:

..
  TODO: add link to internal data storage implementation docs to the `text format type section`
  after DOCS-#2954 is merged

* text format type - file parsing begins from retrieving ``start`` and ``end`` parameters
  from ``parse`` kwargs - these parameters define start and end bytes of data file, that
  should be read in the concrete partition. Using this data and file handle got from
  ``fname``, binary data is read by python ``read`` function. Then resulting data is passed
  into ``pandas.read_*`` function as ``io.BytesIO`` object to get corresponding
  ``pandas.DataFrame`` (we need to do this because Modin partitions internally stores data
  as ``pandas.DataFrame``).

* columnar store type - in this case data chunk to be read is defined by columns names
  passed as ``columns`` parameter as part of ``parse`` kwargs, so no additional action is
  needed and ``fname`` and ``kwargs`` are just passed into ``pandas.read_*`` function (in
  some corner cases ``pyarrow.read_*`` function can be used).

* SQL type - chunking is incorporated in the ``sql`` parameter as part of query, so
  ``parse`` parameters are passed into ``pandas.read_sql`` function without modification.
