pandas on Dask
==============

This section describes usage related documents for the pandas on Dask component of Modin.

Modin uses pandas as a primary memory format of the underlying partitions and optimizes queries
ingested from the API layer in a specific way to this format. Thus, there is no need to care of choosing it
but you can explicitly specify it anyway as shown below.

One of the execution engines that Modin uses is Dask. To enable this engine you should set the following environment variables:

.. code-block:: bash

   export MODIN_ENGINE=dask
   export MODIN_STORAGE_FORMAT=pandas

or turn them on in source code:

.. code-block:: python

   import modin.config as cfg
   cfg.Engine.put('dask')
   cfg.StorageFormat.put('pandas')