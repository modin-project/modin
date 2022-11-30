pandas on Unidist
=================

This section describes usage related documents for the pandas on Unidist component of Modin.

Modin uses pandas as a primary memory format of the underlying partitions and optimizes queries
ingested from the API layer in a specific way to this format. Thus, there is no need to care of choosing it
but you can explicitly specify it anyway as shown below.

One of the execution engines that Modin uses is Unidist. At the moment, modin only supports work with Unidist on MPI.
To enable the pandas on Unidist execution you should set the following environment variables:

.. code-block:: bash

   export MODIN_ENGINE=Unidist
   export UNIDIST_BACKEND=mpi
   export MODIN_STORAGE_FORMAT=pandas

or turn it on in source code:

.. code-block:: python

   import modin.config as cfg
   cfg.Engine.put('ray')
   cfg.StorageFormat.put('pandas')
   import unidist.config as unidist_cfg
   unidist_cfg.Backend.put('mpi')

then you should use mpiexec -n 1 python <script.py> command 
to run your script (please, refer to `Unidist documentation`_ for more information):

.. code-block:: bash

   $ mpiexec -n 1 python script.py

.. _`Unidist documentation`: https://unidist.readthedocs.io/en/latest/using_unidist/unidist_on_mpi.html