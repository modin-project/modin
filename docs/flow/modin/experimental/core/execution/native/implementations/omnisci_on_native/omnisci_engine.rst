:orphan:

.. TODO: all of the info from this file should be moved somewhere else

Column name mangling
''''''''''''''''''''

In ``pandas.DataFrame`` columns might have names not allowed in SQL (e. g.
an empty string). To handle this we simply add '`F_`' prefix to
column names. Index labels are more tricky because they might be non-unique.
Indexes are represented as regular columns, and we have to perform a special
mangling to get valid and unique column names. Demangling is done when we
transform our frame (i.e. its Arrow table) into ``pandas.DataFrame`` format.
