Distributed XGBoost on Modin
============================

Modin provides an implementation of `distributed XGBoost`_ machine learning
algorithm on Modin DataFrames. Please note that this feature is experimental and behavior or
interfaces could be changed.

Install XGBoost on Modin
------------------------

Modin comes with all the dependencies except ``xgboost`` package by default.
Currently, distributed XGBoost on Modin is only supported on the Ray execution engine, therefore, see
the :doc:`installation page </getting_started/installation>` for more information on installing Modin with the Ray engine.
To install ``xgboost`` package you can use ``pip``:

.. code-block:: bash

  pip install xgboost


XGBoost Train and Predict
-------------------------

Distributed XGBoost functionality is placed in ``modin.experimental.xgboost`` module.
``modin.experimental.xgboost`` provides a drop-in replacement API for ``train`` and ``Booster.predict`` xgboost functions.

.. automodule:: modin.experimental.xgboost
  :noindex:
  :members: train

.. autoclass:: modin.experimental.xgboost.Booster
  :noindex:
  :members: predict


ModinDMatrix
------------

Data is passed to ``modin.experimental.xgboost`` functions via a Modin ``DMatrix`` object.

.. automodule:: modin.experimental.xgboost
  :noindex:
  :members: DMatrix

Currently, the Modin ``DMatrix`` supports ``modin.pandas.DataFrame`` only as an input.


A Single Node / Cluster setup
-----------------------------

The XGBoost part of Modin uses a Ray resources by similar way as all Modin functions.

To start the Ray runtime on a single node:

.. code-block:: python

  import ray
  # Look at the Ray documentation with respect to the Ray configuration suited to you most.
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
  # Look at the Ray documentation with respect to the Ray configuration suited to you most.
  ray.init() # Start the Ray runtime for single-node

  import modin.pandas as pd
  import modin.experimental.xgboost as xgb

  # Load iris dataset from sklearn
  iris = datasets.load_iris()

  # Create Modin DataFrames
  X = pd.DataFrame(iris.data)
  y = pd.DataFrame(iris.target)

  # Create DMatrix
  dtrain = xgb.DMatrix(X, y)
  dtest = xgb.DMatrix(X, y)

  # Set training parameters
  xgb_params = {
      "eta": 0.3,
      "max_depth": 3,
      "objective": "multi:softprob",
      "num_class": 3,
      "eval_metric": "mlogloss",
  }
  steps = 20

  # Create dict for evaluation results
  evals_result = dict()

  # Run training
  model = xgb.train(
      xgb_params,
      dtrain,
      steps,
      evals=[(dtrain, "train")],
      evals_result=evals_result
  )

  # Print evaluation results
  print(f'Evals results:\n{evals_result}')

  # Predict results
  prediction = model.predict(dtest)

  # Print prediction results
  print(f'Prediction results:\n{prediction}')



.. _Dataframe: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html
.. _`starting ray`: https://docs.ray.io/en/master/starting-ray.html
.. _`the Iris Dataset`: https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html
.. _`distributed XGBoost`: https://medium.com/intel-analytics-software/distributed-xgboost-with-modin-on-ray-fc17edef7720
