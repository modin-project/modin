Pandas Parsers Module Description
"""""""""""""""""""""""""""""""""
High-Level Module Overview
''''''''''''''''''''''''''

This module houses parser classes (classes that are used for data parsing on the workers)
and util functions for handling parsing results. ``PandasParser`` is base class for parser
classes with pandas storage format, that contains methods common for all child classes. Other
module classes implement ``parse`` function that performs parsing of specific format data
basing on the chunk information computed in the ``modin.core.io`` module. After
chunk data parsing is completed, resulting ``DataFrame``-s will be splitted into smaller
``DataFrame``-s according to ``num_splits`` parameter, data type and number or
rows/columns in the parsed chunk, and then these frames and some additional metadata will
be returned.

.. note:: 
    If you are interested in the data parsing mechanism implementation details, please refer
    to the source code documentation.

Public API
''''''''''

.. automodule:: modin.core.storage_formats.pandas.parsers
    :members:
