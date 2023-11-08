Modin XGBoost module description
""""""""""""""""""""""""""""""""
High-level Module Overview
''''''''''''''''''''''''''

This module holds classes, public interface and internal functions for distributed XGBoost in Modin.

Public classes :py:class:`~modin.experimental.xgboost.Booster`, :py:class:`~modin.experimental.xgboost.DMatrix`
and function :py:func:`~modin.experimental.xgboost.train` provide the user with familiar XGBoost interfaces.
They are located in the ``modin.experimental.xgboost.xgboost`` module.

The internal module ``modin.experimental.xgboost.xgboost.xgboost_ray`` contains the implementation of Modin XGBoost
for the Ray execution engine. This module mainly consists of the Ray actor-class :py:class:`~modin.experimental.xgboost.xgboost_ray.ModinXGBoostActor`,
a function to distribute Modin's partitions between actors :py:func:`~modin.experimental.xgboost.xgboost_ray._assign_row_partitions_to_actors`,
an internal :py:func:`~modin.experimental.xgboost.xgboost_ray._train`/:py:func:`~modin.experimental.xgboost.xgboost_ray._predict`
function used from the public interfaces and additional util functions for computing cluster resources, actor creations etc.

Public interfaces
'''''''''''''''''

:py:class:`~modin.experimental.xgboost.DMatrix` inherits original class ``xgboost.DMatrix`` and overrides
its constructor, which currently supports only `data` and `label` parameters. Both of the parameters must
be ``modin.pandas.DataFrame``, which will be internally unwrapped to lists of delayed objects of Modin's
row partitions using the function :py:func:`~modin.distributed.dataframe.pandas.unwrap_partitions`.

.. autoclass:: modin.experimental.xgboost.DMatrix
  :members:

:py:class:`~modin.experimental.xgboost.Booster` inherits original class ``xgboost.Booster`` and
overrides method ``predict``. The difference from original class interface for ``predict``
method is changing the type of the `data` parameter to :py:class:`~modin.experimental.xgboost.DMatrix`.

.. autoclass:: modin.experimental.xgboost.Booster
    :members:

:py:func:`~modin.experimental.xgboost.train` function has 2 differences from the original ``train`` function - (1) the
data type of `dtrain` parameter is :py:class:`~modin.experimental.xgboost.DMatrix` and (2) a new parameter `num_actors`.

.. autofunction:: modin.experimental.xgboost.train

Internal execution flow on Ray engine
'''''''''''''''''''''''''''''''''''''

Internal functions :py:func:`~modin.experimental.xgboost.xgboost_ray._train` and
:py:func:`~modin.experimental.xgboost.xgboost_ray._predict` work similar to xgboost.


Training
********

1. The data is passed to the :py:func:`~modin.experimental.xgboost.xgboost_ray._train`
   function as a :py:class:`~modin.experimental.xgboost.DMatrix` object. Lists of ``ray.ObjectRef``
   corresponding to row partitions of Modin DataFrames are extracted by iterating over the 
   :py:class:`~modin.experimental.xgboost.DMatrix`. Example:

   .. code-block:: python

     # Extract lists of row partitions from dtrain (DMatrix object)
     X_row_parts, y_row_parts = dtrain
   ..

2. On this step, the parameter `num_actors` is processed. The internal function :py:func:`~modin.experimental.xgboost.xgboost_ray._get_num_actors`
   examines the value provided by the user. In case the value isn't provided, the `num_actors` will be computed using condition that 1 actor should use maximum 2 CPUs.
   This condition was chosen for using maximum parallel workers with multithreaded XGBoost training (2 threads
   per worker will be used in this case).

.. note:: `num_actors` parameter is made available for public function :py:func:`~modin.experimental.xgboost.train` to allow
  fine-tuning for obtaining the best performance in specific use cases.

3. :py:class:`~modin.experimental.xgboost.xgboost_ray.ModinXGBoostActor` objects are created.

4. Data `dtrain` is split between actors evenly. The internal function
   :py:func:`~modin.experimental.xgboost.xgboost_ray._split_data_across_actors` runs assigning row partitions to actors
   using internal function :py:func:`~modin.experimental.xgboost.xgboost_ray._assign_row_partitions_to_actors`.
   This function creates a dictionary in the form: `{actor_rank: ([part_i0, part_i3, ..], [0, 3, ..]), ..}`.

.. note:: :py:func:`~modin.experimental.xgboost.xgboost_ray._assign_row_partitions_to_actors` takes into account IP
  addresses of row partitions of `dtrain` data to minimize excess data transfer.

5. For each :py:class:`~modin.experimental.xgboost.xgboost_ray.ModinXGBoostActor` object ``set_train_data`` method is
   called remotely. This method runs loading row partitions in actor according to the dictionary with partitions
   distribution from previous step. When data is passed to the actor, the row partitions are automatically materialized
   (``ray.ObjectRef`` -> ``pandas.DataFrame``).

6. ``train`` method of :py:class:`~modin.experimental.xgboost.xgboost_ray.ModinXGBoostActor` class object is called remotely. This method
   runs XGBoost training on local data of actor, connects to ``Rabit Tracker`` for sharing training state between
   actors and returns dictionary with `booster` and `evaluation results`.

7. At the final stage results from actors are returned. `booster` and `evals_result` are returned using ``ray.get``
   function from remote actor.


Prediction
**********

1. The data is passed to :py:func:`~modin.experimental.xgboost.xgboost_ray._predict`
   function as a :py:class:`~modin.experimental.xgboost.DMatrix` object.

2. :py:func:`~modin.experimental.xgboost.xgboost_ray._map_predict` function is applied remotely for each partition
   of the data to make a partial prediction.

3. Result ``modin.pandas.DataFrame`` is created from ``ray.ObjectRef`` objects, obtained in the previous step.


Internal API
''''''''''''
.. autoclass:: modin.experimental.xgboost.xgboost_ray.ModinXGBoostActor
  :members:
  :private-members:

.. autofunction:: modin.experimental.xgboost.xgboost_ray._assign_row_partitions_to_actors
.. autofunction:: modin.experimental.xgboost.xgboost_ray._train
.. autofunction:: modin.experimental.xgboost.xgboost_ray._predict
.. autofunction:: modin.experimental.xgboost.xgboost_ray._get_num_actors
.. autofunction:: modin.experimental.xgboost.xgboost_ray._split_data_across_actors
.. autofunction:: modin.experimental.xgboost.xgboost_ray._map_predict
