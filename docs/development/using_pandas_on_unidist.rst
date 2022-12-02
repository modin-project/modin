pandas on Unidist
=================

This section describes usage related documents for the pandas on Unidist component of Modin.

Modin uses pandas as a primary memory format of the underlying partitions and optimizes queries
ingested from the API layer in a specific way to this format. Thus, there is no need to care of choosing it
but you can explicitly specify it anyway as shown below.

One of the execution engines that Modin uses is Unidist. Currently, Modin only supports Unidist on MPI backend.
To enable the pandas on Unidist execution using MPI backend you should set the following environment variables:

.. code-block:: bash

   export MODIN_ENGINE=unidist
   export MODIN_STORAGE_FORMAT=pandas
   export UNIDIST_BACKEND=mpi

or turn it on in source code:

.. code-block:: python

   import modin.config as modin_cfg
   import unidist.config as unidist_cfg

   modin_cfg.Engine.put('unidist')
   modin_cfg.StorageFormat.put('pandas')
   unidist_cfg.Backend.put('mpi')

To run a python application you should use ``mpiexec -n 1 python <script.py>`` command.

.. code-block:: bash

   $ mpiexec -n 1 python script.py

For more information on how to run a python application with unidist on MPI backend
please refer to `Unidist on MPI`_ section of the unidist documentation.

.. _`Unidist on MPI`: https://unidist.readthedocs.io/en/latest/using_unidist/unidist_on_mpi.html