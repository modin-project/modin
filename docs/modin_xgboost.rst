Distributed XGBoost on Modin (experimental)
===========================================

Modin provides own effective implementation of a distributed XGBoost machine learning 
algorithm on Modin DataFrames. Please note that this feature is experimental and behavior or 
interfaces could be changed.

Install XGBoost on Modin
------------------------

Modin comes with all the dependencies exclude of ``xgboost`` package by default.
Currently, the distributed XGBoost on Modin supports the Ray backend only, therefore, see
the `installation page`_ for more information on installing Modin with the Ray backend.
To install ``xgboost`` package you can use ``pip``:

.. code-block:: bash

  pip install xgboost


Learning Interfaces
-------------------

Distributed XGBoost functionality is placed in ``modin.experimental.xgboost`` module.
``modin.experimental.xgboost`` provides a xgboost-like API for ``train`` and ``predict`` functions.

``train`` has all arguments of ``xgboost.train`` function exclude the `evals_result`
parameter which is returned as part of function return value instead of argument.

``predict`` is separate function unlike ``xgboost.Booster.predict`` which uses an additional argument
``model``. ``model`` could be ``xgboost.Booster`` or output of ``modin.experimental.xgboost`` function.

Both of function have additional parameters `nthread` and `evenly_data_distribution`.
`nthread` sets number of threads to use per node in cluster.
`evenly_data_distribution` sets rule of distribution data between nodes in cluster.


Type of input data
------------------

Data is passed to ``modin.experimental.xgboost`` functions via a ``ModinDMatrix`` object.

The ``ModinDMatrix`` stores data as Modin DataFrames internally. 

Currently, the ``ModinDMatrix`` supports ``modin.pandas.DataFrame`` only as an input.


A Single Node / Cluster setup
-----------------------------

The XGBoost part of Modin uses a Ray resources by similar way as all Modin functions.

To start the Ray runtime on a single node:

.. code-block:: python

  import ray
  ray.init()

If you already had the Ray cluster you can connect to it by next way:

.. code-block:: python

  import ray
  ray.init(address='auto')

A detailed information about initializing the Ray runtime you can find in `starting ray`_  page.


Usage example
-------------

In example below we train XGBoost model using `the Iris Dataset`_ and get prediction on the same data.
All processing will be in a `single node` mode.

.. code-block:: python

  from sklearn import datasets
  
  import ray
  ray.init() # Start the Ray runtime for single-node
  
  import modin.pandas as pd
  import modin.experimental.xgboost as xgb
  
  # Load iris dataset from sklearn
  iris = datasets.load_iris()
  
  # Create Modin DataFrames
  X = pd.DataFrame(iris.data)
  y = pd.DataFrame(iris.target)
  
  # Create ModinDMatrix
  dtrain = xgb.ModinDMatrix(X, y)
  dtest = xgb.ModinDMatrix(X, y)
  
  # Set training parameters
  xgb_params = {
      "eta": 0.3,
      "max_depth": 3,
      "objective": "multi:softprob",
      "num_class": 3,
      "eval_metric": "mlogloss",
  }
  steps = 20
  
  # Run training
  model = xgb.train(
      xgb_params,
      dtrain,
      steps,
      evals=[(dtrain, "train")]
  )
  
  # Save for some usage
  evals_result = model["history"]
  booster = model["booster"]
  
  # Predict results
  prediction = xgb.predict(model, dtest)


Modes of a data distribution
----------------------------

Modin XGBoost provides two approaches for an internal data ditribution which could be
switched by `evenly_data_distribution` parameter of ``train/predict`` functions:

* `evenly_data_distribution` = `True`: in this case the input data of ``train/predict``
  functions will be distributed evenly between nodes in a cluster to ensure evenly utilization of nodes (default behavior).

* `evenly_data_distribution` = `False` :  in this case partitions of input data of ``train/predict``
  functions will not transfer between nodes in cluster in case empty nodes is <10%,
  if portion of empty nodes is ≥10% evenly data distribution will be applied.
  This method provides minimal data transfers between nodes but doesn't guarantee effective utilization of nodes.
  Most effective in case when all cluster nodes are occupied by data.


.. _Dataframe: https://pandas.pydata.org/pandas-docs/version/0.23.4/generated/pandas.DataFrame.html
.. _`installation page`: installation.html
.. _`starting ray`: https://docs.ray.io/en/master/starting-ray.html
.. _`the Iris Dataset`: https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html
