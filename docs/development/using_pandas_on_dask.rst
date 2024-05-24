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

If you want to use a single node, just change the Modin Engine to Dask and 
continue working with the Modin Dataframe as if it were a Pandas Dataframe.
You don't even have to initialize the Dask Client, because Modin will do it 
yourself or use the current one if it is already initialized:

.. code-block:: python

  import modin.pandas as pd
  import modin.config as modin_cfg

  modin_cfg.Engine.put("dask")
  df = pd.read_parquet("s3://my-bucket/big.parquet")

.. note:: In previous versions of Modin, you had to initialize Dask before importing Modin. As of Modin 0.9.0, This is no longer the case.

Using Modin on Dask Clusters
----------------------------

If you want to use clusters of many machines, you don't need to do any additional steps.
Just initialize a Dask Client on your cluster and use Modin as you would on a single node.
As long as Dask Client is initialized before any dataframes are created, Modin
will be able to connect to and use the Dask Cluster.

.. code-block:: python

  from distributed import Client
  import modin.pandas as pd
  import modin.config as modin_cfg
  
  # Please define your cluster here
  cluster = ...
  client = Client(cluster)

  modin_cfg.Engine.put("dask")
  df = pd.read_parquet("s3://my-bucket/big.parquet")

To get more ways to deploy and run Dask clusters, visit the `Deploying Dask Clusters page`_.

How Modin uses Dask
-------------------

Modin has a layered architecture, and the core abstraction for data manipulation
is the Modin Dataframe, which implements a novel algebra that enables Modin to
handle all of pandas (see Modin's documentation_ for more on the architecture).
Modin's internal dataframe object has a scheduling layer that is able to partition
and operate on data with Dask.

Conversion to and from Modin from Dask Dataframe
------------------------------------------------

Modin DataFrame can be converted to/from Dask Dataframe with no-copy partition conversion.
This allows you to take advantage of both Dask and Modin libraries for maximum performance.

.. code-block:: python

  import modin.pandas as pd
  import modin.config as modin_cfg
  from modin.pandas.io import to_dask, from_dask

  modin_cfg.Engine.put("dask")
  df = pd.read_parquet("s3://my-bucket/big.parquet")

  # Convert Modin to Dask Dataframe
  dask_df = to_dask(df)
  
  # Convert Dask to Modin Dataframe
  modin_df = from_dask(dask_df)

.. _Deploying Dask Clusters page: https://docs.dask.org/en/stable/deploying.html
.. _documentation: https://modin.readthedocs.io/en/latest/development/architecture.html