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


Running on a GPU
----------------

Prerequisites:

* HDK's GPU mode is currently supported on Linux and Intel GPU only.
* HDK supports Gen9 architecture and higher (including Xe & Arc).
* HDK's GPU mode requires proper driver installation. Follow this guide_ to set up your system. Make sure to install the compute runtime packages: ``intel-opencl-icd``, ``intel-level-zero-gpu``, ``level-zero``.
* Make sure your GPU is visible and accessible.

.. note::
   You can use ``hwinfo`` and ``clinfo`` utilities to verify the driver installation and device accessibility.

HDK supports a heterogeneous execution mode (experimental) that is disabled by default in Modin. Starting with pyHDK version 0.7 Modin can run the workload on Intel GPU.
Run on a GPU via ``MODIN_HDK_LAUNCH_PARAMETERS="cpu_only=0" python <your-script.py>``.

.. _HDK: https://github.com/intel-ai/hdk
.. _guide: https://dgpu-docs.intel.com/driver/installation.html