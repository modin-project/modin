OmniSci
=======

This section describes usage related documents for the OmniSciDB-based engine of Modin.

This engine uses analytical database OmniSciDB_ to obtain high single-node scalability for
specific set of dataframe operations.
To enable this engine you could set the following environment variables:

.. code-block:: bash

   export MODIN_ENGINE=native
   export MODIN_STORAGE_FORMAT=omnisci
   export MODIN_EXPERIMENTAL=true

or turn it on in source code:

.. code-block:: python

   import modin.config as cfg
   cfg.Engine.put('native')
   cfg.StorageFormat.put('omnisci')
   cfg.IsExperimental.put(True)

To enable ``OmniSci`` engine launch export :

.. code-block:: bash

   export MODIN_STORAGE_FORMAT=omnisci

or use it in your code:

.. code-block:: python

   import modin.config as cfg
   cfg.StorageFormat.put('omnisci')

.. _OmnisciDB: https://www.omnisci.com/platform/omniscidb