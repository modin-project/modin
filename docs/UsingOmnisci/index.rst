OmniSci
=======

This section describes usage related documents for the OmniSciDB-based engine of Modin.

This engine uses analytical database OmniSciDB_ to obtain high single-node scalability for
specific set of dataframe operations.
To enable this engine you could set the following environment variables:

.. code-block:: bash

   export MODIN_ENGINE=native
   export MODIN_BACKEND=omnisci
   export MODIN_EXPERIMENTAL=true

or turn it on in source code:

.. code-block:: python

   import modin.config as cfg
   cfg.Engine.put('native')
   cfg.Backend.put('omnisci')
   cfg.IsExperimental.put(True)


.. _OmnisciDB: https://www.omnisci.com/platform/omniscidb
