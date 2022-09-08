HDK
===

This section describes usage related documents for the HDK-based engine of Modin.

This engine uses the HDK_ library to obtain high single-node scalability for
specific set of dataframe operations.
To enable this engine you can set the following environment variable:

.. code-block:: bash

   export MODIN_STORAGE_FORMAT=hdk

or use it in your code:

.. code-block:: python

   import modin.config as cfg
   cfg.StorageFormat.put('hdk')

Since HDK is run through its native engine, Modin automatically sets ``MODIN_ENGINE=Native`` and you might not specify it explicitly.
If for some reasons ``Native`` engine is explicitly set using ``modin.config`` or
``MODIN_ENGINE`` environment variable, make sure you also tell Modin that
``Experimental`` mode is turned on (``export MODIN_EXPERIMENTAL=true`` or 
``cfg.IsExperimental.put(True)``) otherwise the following error occurs:

.. code-block:: bash

   FactoryNotFoundError: HDK on Native is only accessible through the experimental API.
   Run `import modin.experimental.pandas as pd` to use HDK on Native.


.. note::
   If you encounter ``LLVM ERROR: inconsistency in registered CommandLine options`` error when using HDK,
   please refer to the respective section in :doc:`Troubleshooting </getting_started/troubleshooting>` page to avoid the issue.

.. _HDK: https://github.com/intel-ai/hdk