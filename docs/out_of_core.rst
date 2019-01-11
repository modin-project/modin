Out of Core in Modin
====================

If you are working with very large files or would like to exceed your memory, you may
change the primary location of the DataFrame. If you would like to exceed memory, you
can use your disk as an overflow for the memory. This API is experimental in the context
of Modin. Please let us know what you think!

Install Modin out of core
-------------------------

As we mentioned in the `installation page`_, we have set up a select dependency set for
users who want to use Modin out of core. It can be installed with pip:

.. code-block:: bash

  pip install "modin[out_of_core]"

This will ensure that you have all of the required dependencies for Modin to run out of
core.

Starting Modin with out of core enabled
---------------------------------------

Out of core is detected from an environment variable set in bash.

.. code-block:: bash

   export MODIN_OUT_OF_CORE=true

We also set up a way to tell Modin how much memory you'd like to use. Currently, this
only accepts the number of bytes. This can only exceed your memory if you have enabled
``MODIN_OUT_OF_CORE``.

**Warning: Make sure you have enough space in your disk for however many bytes you**
**request for your DataFrame**

Here is how you set ``MODIN_MEMORY``:

.. code-block:: bash

  export MODIN_MEMORY=200000000000 # Set the number of bytes to 200GB

This limits the amount of memory that Modin can use.

Running an example with out of core
-----------------------------------

.. _`installation page`: installation.html