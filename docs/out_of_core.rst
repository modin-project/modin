Out of Core in Modin (experimental)
===================================

If you are working with very large files or would like to exceed your memory, you may
change the primary location of the `DataFrame`_. If you would like to exceed memory, you
can use your disk as an overflow for the memory. This API is experimental in the context
of Modin. Please let us know what you think!

Install Modin out of core
-------------------------

Modin now comes with all the dependencies for out of core functionality by default! See
the `installation page`_ for more information on installing Modin.

Starting Modin with out of core enabled
---------------------------------------

Out of core is detected from an environment variable set in bash.

.. code-block:: bash

   export MODIN_OUT_OF_CORE=true

We also set up a way to tell Modin how much memory you'd like to use. Currently, this
only accepts the number of bytes. This can only exceed your memory if you have enabled
``MODIN_OUT_OF_CORE``.

[Optional]: Set a limit on the out of core space for Modin
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

**Warning: Make sure you have enough space in your disk for however many bytes you**
**request for your DataFrame**

This limits the amount of memory that Modin can use.

Here is how you set ``MODIN_MEMORY``:

.. code-block:: bash

  export MODIN_MEMORY=200000000000 # Set the number of bytes to 200GB


**The default for Modin is 8x the memory on the machine.**

Running an example with out of core
-----------------------------------

Before you run this, please make sure you follow the instructions listed above.

.. code-block:: python

  import modin.pandas as pd
  import numpy as np
  frame_data = np.random.randint(0, 100, size=(2**20, 2**8)) # 2GB each
  df = pd.DataFrame(frame_data).add_prefix("col")
  big_df = pd.concat([df for _ in range(20)]) # 20x2GB frames
  print(big_df)
  nan_big_df = big_df.isna() # The performance here represents a simple map
  print(big_df.apply(lambda col: col.sum())) # apply along an entire axis (columns in this case)

This example creates a 40GB DataFrame from 20 identical 2GB DataFrames and performs
various operations on them. Feel free to play around with this code and let us know what
you think!

.. _Dataframe: https://pandas.pydata.org/pandas-docs/version/0.23.4/generated/pandas.DataFrame.html
.. _`installation page`: installation.html
