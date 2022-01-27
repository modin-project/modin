pandas on Ray
=============

This section describes usage related documents for the pandas on Ray component of Modin.

Modin uses pandas as a primary memory format of the underlying partitions and optimizes queries
ingested from the API layer in a specific way to this format. Thus, there is no need to care of choosing it
but you can explicitly specify it anyway as shown below.

One of the execution engines that Modin uses is Ray. If you have Ray installed in your system,
Modin also uses it by default to distribute computations.

If you want to be explicit, you could set the following environment variables:

.. code-block:: bash

   export MODIN_ENGINE=ray
   export MODIN_STORAGE_FORMAT=pandas

or turn it on in source code:

.. code-block:: python

   import modin.config as cfg
   cfg.Engine.put('ray')
   cfg.StorageFormat.put('pandas')
