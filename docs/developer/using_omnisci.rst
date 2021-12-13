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

Since OmniSci is run through its native engine Modin itself sets ``MODIN_ENGINE=Native``
and you might not specify it explicitly.

.. note::
   If you encounter ``LLVM ERROR: inconsistency in registered CommandLine options`` error when using OmniSci,
   please refer to the respective section in :doc:`Troubleshooting </developer/troubleshooting>` page to avoid the issue.

.. _OmnisciDB: https://www.omnisci.com/platform/omniscidb
