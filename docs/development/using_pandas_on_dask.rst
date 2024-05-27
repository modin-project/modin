pandas on Dask
==============

This section describes usage related documents for the pandas on Dask component of Modin.

Modin uses pandas as a primary memory format of the underlying partitions and optimizes queries
ingested from the API layer in a specific way to this format. Thus, there is no need to care of choosing it
but you can explicitly specify it anyway as shown below.

One of the execution engines that Modin uses is Dask. To enable the pandas on Dask execution you should set the following environment variables:

.. code-block:: bash

   export MODIN_ENGINE=dask
   export MODIN_STORAGE_FORMAT=pandas

or turn them on in source code:

.. code-block:: python

   import modin.config as cfg
   cfg.Engine.put('dask')
   cfg.StorageFormat.put('pandas')

Using Modin on Dask locally
---------------------------

If you want to run Modin on Dask locally using a single node, just set Modin engine to ``Dask`` and 
continue working with a Modin DataFrame as if it was a pandas DataFrame.
You can either initialize a Dask client on your own and Modin connects to the existing Dask cluster or
allow Modin itself to initialize a Dask client.

.. code-block:: python

  import modin.pandas as pd
  import modin.config as modin_cfg

  modin_cfg.Engine.put("dask")
  df = pd.DataFrame(...)

Using Modin on Dask in a Cluster
--------------------------------

If you want to run Modin on Dask in a cluster, you should set up a Dask cluster and initialize a Dask client.
Once the Dask client is initialized, Modin will be able to connect to it and use the Dask cluster.

.. code-block:: python

  from distributed import Client
  import modin.pandas as pd
  import modin.config as modin_cfg
  
  # Define your cluster here
  cluster = ...
  client = Client(cluster)

  modin_cfg.Engine.put("dask")
  df = pd.DataFrame(...)

To get more information on how to deploy and run a Dask cluster, visit the `Deploy Dask Clusters`_ page.

Conversion between Modin DataFrame and Dask DataFrame
-----------------------------------------------------

Modin DataFrame can be converted to/from Dask DataFrame with no-copy partition conversion.
This allows you to take advantage of both Modin and Dask libraries for maximum performance.

.. code-block:: python

  import modin.pandas as pd
  import modin.config as modin_cfg
  from modin.pandas.io import to_dask, from_dask

  modin_cfg.Engine.put("dask")
  df = pd.DataFrame(...)

  # Convert Modin to Dask DataFrame
  dask_df = to_dask(df)
  
  # Convert Dask to Modin DataFrame
  modin_df = from_dask(dask_df)

.. _Deploy Dask Clusters: https://docs.dask.org/en/stable/deploying.html
