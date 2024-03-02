Out-of-memory data with Modin
=============================

.. note::
  | *Estimated Reading Time: 10 minutes*
  
When using pandas, you might run into a memory error if you are working with large datasets that cannot fit in memory or perform certain memory-intensive operations (e.g., joins). 

Modin solves this problem by spilling over to disk, in other words, it uses your disk as an overflow for memory so that you can work with datasets that are too large to fit in memory. By default, Modin leverages out-of-core methods to handle datasets that don't fit in memory for both Ray and Dask engines.

.. note::
  Object spilling is disabled in a multi-node Ray cluster by default. To enable object spilling
  use `Ray instruction <https://docs.ray.io/en/latest/ray-core/objects/object-spilling.html#cluster-mode>`_.


Motivating Example: Memory error with pandas
--------------------------------------------

pandas makes use of in-memory data structures to store and operate on data, which means that if you have a dataset that is too large to fit in memory, it will cause an error on pandas. As an example, let's creates a 80GB DataFrame by appending together 40 different 2GB DataFrames. 

.. code-block:: python

  import pandas
  import numpy as np
  df = pandas.concat([pandas.DataFrame(np.random.randint(0, 100, size=(2**20, 2**8))) for _ in range(40)]) # Memory Error!

When we run this on a laptop with 32GB of RAM, pandas will run out of memory and throw an error (e.g., :code:`MemoryError` , :code:`Killed: 9`). 

The `pandas documentation <https://pandas.pydata.org/pandas-docs/stable/user_guide/scale.html>`_ has a great section on recommendations for scaling your analysis to these larger datasets. However, this generally involves loading in less data or rewriting your pandas code to process the data in smaller chunks. 

Operating on out-of-memory data with Modin
------------------------------------------

In order to work with data that exceeds memory constraints, you can use Modin to handle these large datasets.

.. code-block:: python

  import modin.pandas as pd
  import numpy as np
  df = pd.concat([pd.DataFrame(np.random.randint(0, 100, size=(2**20, 2**8))) for _ in range(40)]) # 40x2GB frames -- Working!
  df.info()

Not only does Modin let you work with datasets that are too large to fit in memory, we can perform various operations on them without worrying about memory constraints. 

Advanced: Configuring out-of-core settings
------------------------------------------

.. why would you want to disable out of core?

By default, out-of-core functionality is enabled by the compute engine selected. 
To disable it, start your preferred compute engine with the appropriate arguments. For example:

.. code-block:: python

  import modin.pandas as pd
  import ray

  ray.init(_plasma_directory="/tmp")  # setting to disable out of core in Ray
  df = pd.read_csv("some.csv")

If you are using Dask, you have to modify local configuration files. Visit the
Dask documentation_ on object spilling for more details.


.. _documentation: https://distributed.dask.org/en/latest/worker.html#memory-management
