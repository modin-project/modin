OmniSci
=======

To enable ``OmniSci`` engine launch export :

.. code-block:: bash

   export MODIN_STORAGE_FORMAT=omnisci

or use it in your code:

.. code-block:: python

   import modin.config as cfg
   cfg.StorageFormat.put('omnisci')

.. _OmnisciDB: https://www.omnisci.com/platform/omniscidb