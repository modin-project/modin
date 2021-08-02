OmniSci
=======

This section describes usage related documents for the OmniSciDB-based engine of Modin.

This engine uses analytical database OmniSciDB_ to obtain high single-node scalability for
specific set of dataframe operations.
To enable this engine you could set the following environment variables:

.. code-block:: bash

   export MODIN_ENGINE=ray
   export MODIN_BACKEND=omnisci

or turn it on in source code:

.. code-block:: python

   import modin.experimental.pandas as pd
   import modin.config as cfg
   cfg.Engine.put('ray')
   cfg.Backend.put('omnisci')


.. _OmnisciDB: https://www.omnisci.com/platform/omniscidb
