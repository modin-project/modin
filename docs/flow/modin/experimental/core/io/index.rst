:orphan:

Experimental IO Module Description
""""""""""""""""""""""""""""""""""

The module is used mostly for storing experimental utils and
dispatcher classes for reading/writing files of different formats.

Submodules Description
''''''''''''''''''''''

* text - directory for storing all text file format dispatcher classes

  * format/feature specific dispatchers: ``csv_glob_dispatcher.py``,
    ``custom_text_dispatcher.py``.

* sql - directory for storing SQL dispatcher class

  * format/feature specific dispatchers: ``sql_dispatcher.py``

* pickle - directory for storing Pickle dispatcher class

  * format/feature specific dispatchers: ``pickle_dispatcher.py``

Public API
''''''''''

.. automodule:: modin.experimental.core.io
    :members:
