pandas on Python
================

This section describes usage related documents for the pandas on Python component of Modin.

Modin uses pandas as a primary memory format of the underlying partitions and optimizes queries
ingested from the API layer in a specific way to this format. Thus, there is no need to care of choosing it
but you can explicitly specify it anyway as shown below.

One of the execution engines that Modin uses is Python. This engine is sequential and it's used for the debug purposes.
To enable the pandas on Python execution you should set the following environment variables:

.. code-block:: bash

   export MODIN_ENGINE=python
   export MODIN_STORAGE_FORMAT=pandas

or turn a debug mode on:

.. code-block:: bash

   export MODIN_DEBUG=True
   export MODIN_STORAGE_FORMAT=pandas

or do the same in source code:

.. code-block:: python

   import modin.config as cfg
   cfg.Engine.put('python')
   cfg.StorageFormat.put('pandas')

.. code-block:: python

   import modin.config as cfg
   cfg.IsDebug.put(True)
   cfg.StorageFormat.put('pandas')