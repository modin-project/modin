:orphan:

Pyarrow Parsers Module Description
""""""""""""""""""""""""""""""""""
This module houses parser classes that are responsible for data parsing on the workers for the Pyarrow backend.
Parsers for pyarrow backends follow an interface of :doc:`pandas backend parsers </flow/modin/backends/pandas/parsers>`:
parser class of every file format implements ``parse`` method, which parses the specified part
of the file and builds Pyarrow tables from the parsed data, based on the specified chunk size and number of splits.
The resulted Pyarrow tables will be used as a partitions payload in the ``PyarrowOnRayFrame``.
