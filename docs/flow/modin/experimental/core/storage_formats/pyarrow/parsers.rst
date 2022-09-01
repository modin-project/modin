Experimental PyArrow Parsers Module Description
"""""""""""""""""""""""""""""""""""""""""""""""

This module houses parser classes that are responsible for data parsing on the workers for the PyArrow storage format.
Parsers for PyArrow storage formats follow an interface of :doc:`pandas format parsers </flow/modin/core/storage_formats/pandas/parsers>`:
parser class of every file format implements ``parse`` method, which parses the specified part
of the file and builds PyArrow tables from the parsed data, based on the specified chunk size and number of splits.
The resulted PyArrow tables will be used as a partitions payload in the :py:class:`~modin.experimental.core.execution.ray.implementations.pyarrow_on_ray.dataframe.dataframe.PyarrowOnRayDataframe`.

Public API
''''''''''

.. automodule:: modin.experimental.core.storage_formats.pyarrow.parsers
    :members:

