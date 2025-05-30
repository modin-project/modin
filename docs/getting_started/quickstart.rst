Getting Started
===============

.. note:: 
  | *Estimated Reading Time: 10 minutes*
  | You can follow along this tutorial in a Jupyter notebook `here <https://github.com/modin-project/modin/tree/main/examples/quickstart.ipynb>`_. 

.. toctree::
    :hidden:
    :maxdepth: 4
    
    10-min Quickstart Guide <self>
    installation
    using_modin/using_modin
    why_modin/why_modin
    examples
    faq
    troubleshooting

.. meta::
    :description lang=en:
        Introduction to Modin.

Quick Start Guide
-----------------

To install the most recent stable release for Modin run the following:

.. code-block:: bash

  pip install "modin[all]" 

For further instructions on how to install Modin with conda or for specific platforms 
or engines, see our detailed `installation guide <../getting_started/installation.html>`_.

Modin acts as a drop-in replacement for pandas so you simply have to replace the import 
of pandas with the import of Modin as follows to speed up your pandas workflows:

.. code-block:: bash

  # import pandas as pd
  import modin.pandas as pd

Example: Instant Scalability with No Extra Effort
-------------------------------------------------

When working on large datasets, pandas becomes painfully slow or :doc:`runs out of memory</getting_started/why_modin/out_of_core>`. Modin automatically scales up your 
pandas workflows by parallelizing the dataframe operations, so that you can more 
effectively leverage the compute resources available.

For the purpose of demonstration, we will load in modin as ``pd`` and pandas as 
``pandas``.

.. code-block:: python

  import modin.pandas as pd
  import pandas

  #############################################
  ### For the purpose of timing comparisons ###
  #############################################
  import time
  import ray
  # Look at the Ray documentation with respect to the Ray configuration suited to you most.
  ray.init()
  #############################################

In this toy example, we look at the NYC taxi dataset, which is around 200MB in size. You can download `this dataset <https://modin-datasets.intel.com/testing/yellow_tripdata_2015-01.csv>`_ to run the example locally.

.. code-block:: python

  # This may take a few minutes to download
  import urllib.request
  dataset_url = "https://modin-datasets.intel.com/testing/yellow_tripdata_2015-01.csv"
  urllib.request.urlretrieve(dataset_url, "taxi.csv")  

Faster Data Loading with ``read_csv``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python
   
  start = time.time()

  pandas_df = pandas.read_csv(dataset_url, parse_dates=["tpep_pickup_datetime", "tpep_dropoff_datetime"], quoting=3)

  end = time.time()
  pandas_duration = end - start
  print("Time to read with pandas: {} seconds".format(round(pandas_duration, 3)))

By running the same command ``read_csv`` with Modin, we generally get around 4X speedup 
for loading in the data in parallel. 

.. code-block:: python

  start = time.time()

  modin_df = pd.read_csv(dataset_url, parse_dates=["tpep_pickup_datetime", "tpep_dropoff_datetime"], quoting=3)

  end = time.time()
  modin_duration = end - start
  print("Time to read with Modin: {} seconds".format(round(modin_duration, 3)))

  print("Modin is {}x faster than pandas at `read_csv`!".format(round(pandas_duration / modin_duration, 2)))

Faster ``concat`` across multiple dataframes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Our previous ``read_csv`` example operated on a relatively small dataframe. In the 
following example, we duplicate the same taxi dataset 100 times and then concatenate 
them together, resulting in a dataset around 19GB in size.

.. code-block:: python

  start = time.time()

  big_pandas_df = pandas.concat([pandas_df for _ in range(25)])

  end = time.time()
  pandas_duration = end - start
  print("Time to concat with pandas: {} seconds".format(round(pandas_duration, 3)))

.. code-block:: python

  start = time.time()

  big_modin_df = pd.concat([modin_df for _ in range(25)])

  end = time.time()
  modin_duration = end - start
  print("Time to concat with Modin: {} seconds".format(round(modin_duration, 3)))

  print("Modin is {}x faster than pandas at `concat`!".format(round(pandas_duration / modin_duration, 2)))

Modin speeds up the ``concat`` operation by more than 60X, taking less than a second to 
create the large dataframe, while pandas took close to a minute.


Faster ``apply`` over a single column
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The performance benefits of Modin become apparent when we operate on large 
gigabyte-scale datasets. Let's say we want to round up values 
across a single column via the ``apply`` operation. 

.. code-block:: python

  start = time.time()
  rounded_trip_distance_pandas = big_pandas_df["trip_distance"].apply(round)

  end = time.time()
  pandas_duration = end - start
  print("Time to apply with pandas: {} seconds".format(round(pandas_duration, 3)))

.. code-block:: python
  
  start = time.time()

  rounded_trip_distance_modin = big_modin_df["trip_distance"].apply(round)

  end = time.time()
  modin_duration = end - start
  print("Time to apply with Modin: {} seconds".format(round(modin_duration, 3)))

  print("Modin is {}x faster than pandas at `apply` on one column!".format(round(pandas_duration / modin_duration, 2)))

Modin is more than 30X faster at applying a single column of data, operating on 130+ 
million rows in a second.

In short, Modin provides orders of magnitude speed up over pandas for a variety of operations out of the box. 

.. figure:: ../img/quickstart_speedup.svg
   :align: center

Summary
-------

Hopefully, this tutorial demonstrated how Modin delivers significant speedup on pandas 
operations without the need for any extra effort. Throughout example, we moved from 
working with 100MBs of data to 20GBs of data all without having to change anything or 
manually optimize our code to achieve the level of scalable performance that Modin 
provides.

Note that in this quickstart example, we've only shown ``read_csv``, ``concat``, 
``apply``, but these are not the only pandas operations that Modin optimizes for. In 
fact, Modin covers `more than 90\% of the pandas API <https://github.com/modin-project/modin/blob/main/README.md#pandas-api-coverage>`_, yielding considerable speedups for 
many common operations.
