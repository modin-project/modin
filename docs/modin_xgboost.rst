Distributed XGBoost on Modin (experimental)
===========================================

Modin provides an implementation of distributed XGBoost machine learning 
algorithm on Modin DataFrames. Please note that this feature is experimental and behavior or 
interfaces could be changed.

Install XGBoost on Modin
------------------------

Modin comes with all the dependencies except ``xgboost`` package by default.
Currently, distributed XGBoost on Modin is only supported on the Ray backend, therefore, see
the :doc:`installation page </installation>` for more information on installing Modin with the Ray backend.
To install ``xgboost`` package you can use ``pip``:

.. code-block:: bash

  pip install xgboost


XGBoost Train and Predict
-------------------------

Distributed XGBoost functionality is placed in ``modin.experimental.xgboost`` module.
``modin.experimental.xgboost`` provides a xgboost-like API for ``train`` and ``predict`` functions.

.. automodule:: modin.experimental.xgboost
  :members: train

``train`` has all arguments of the ``xgboost.train`` function except for ``evals_result``
parameter which is returned as part of function return value instead of argument.

.. automodule:: modin.experimental.xgboost
  :noindex:
  :members: predict

``predict`` is similar to ``xgboost.Booster.predict`` with an additional argument,
``model``.


ModinDMatrix
------------

Data is passed to ``modin.experimental.xgboost`` functions via a ``ModinDMatrix`` object.

.. automodule:: modin.experimental.xgboost
  :noindex:
  :members: ModinDMatrix

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

Modin XGBoost provides three approaches for an internal data ditribution which could be
switched by `distribution_type` parameter of ``train/predict`` functions. Types of distribution
are described in ``DistributionType`` enum.

.. automodule:: modin.experimental.xgboost
  :noindex:
  :members: DistributionType

:note: In case ``distribution_type`` parameter of ``train/predict`` functions is ``DistributionType.LOCALLY``,
partitions of input data for those functions will not transfer between nodes in cluster in case empty nodes is <10%,
if portion of empty nodes is ≥10%, evenly data distribution will be applied.
This method provides minimal data transfers between nodes but doesn't guarantee effective utilization of nodes.
Most effective in case when all cluster nodes are occupied by data.


.. _Dataframe: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html
.. _`starting ray`: https://docs.ray.io/en/master/starting-ray.html
.. _`the Iris Dataset`: https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html
