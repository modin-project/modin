OmniSci
=======

This section describes usage related documents for the OmniSciDB-based engine of Modin.

This engine uses analytical database OmniSciDB_ to obtain high single-node scalability for
specific set of dataframe operations.
To enable this engine you can set the following environment variable:

.. code-block:: bash

   export MODIN_STORAGE_FORMAT=omnisci

or use it in your code:

.. code-block:: python

   import modin.config as cfg
   cfg.StorageFormat.put('omnisci')

Since OmniSci is run through its native engine, Modin automatically sets ``MODIN_ENGINE=Native`` and you might not specify it explicitly.
If for some reasons ``Native`` engine is explicitly set using ``modin.config`` or
``MODIN_ENGINE`` environment variable, make sure you also tell Modin that
``Experimental`` mode is turned on (``export MODIN_EXPERIMENTAL=true`` or ``cfg.IsExperimental.put(True)``) otherwise following error occurs:

.. code-block:: bash

   FactoryNotFoundError: Omnisci on Native is only accessible through the experimental API.
   Run `import modin.experimental.pandas as pd` to use Omnisci on Native.


.. note::
   If you encounter ``LLVM ERROR: inconsistency in registered CommandLine options`` error when using OmniSci,
   please refer to the respective section in :doc:`Troubleshooting </development/troubleshooting>` page to avoid the issue.

.. _OmnisciDB: https://www.omnisci.com/platform/omniscidb