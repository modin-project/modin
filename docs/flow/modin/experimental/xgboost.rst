Modin XGBoost module description
""""""""""""""""""""""""""""""""
High-level Module Overview
''''''''''''''''''''''''''

This module holds classes, public interface and internal functions for distributed XGBoost in Modin.

Public classes :py:class:`~modin.experimental.xgboost.Booster`, :py:class:`~modin.experimental.xgboost.DMatrix`
and function :py:func:`~modin.experimental.xgboost.train` provide the user with familiar XGBoost interfaces.
They are located in the ``modin.experimental.xgboost.xgboost`` module.

The internal module ``modin.experimental.xgboost.xgboost.xgboost_ray`` contains the implementation of Modin XGBoost
for the Ray backend. This module mainly consists of the Ray actor-class :py:class:`~modin.experimental.xgboost.xgboost_ray.ModinXGBoostActor`,
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
overrides method ``predict``. The main differences from original class interface for ``predict``
method are: (1) changing the type of the `data` parameter to :py:class:`~modin.experimental.xgboost.DMatrix`, and (2)
a new parameter `num_actors`, which specifies the number of actors to run for prediction.

.. autoclass:: modin.experimental.xgboost.Booster
    :members:

:py:func:`~modin.experimental.xgboost.train` function (similar to ``predict`` method of
:py:class:`~modin.experimental.xgboost.Booster`) has 2 differences from the original ``train`` function - (1) the
data type of `dtrain` parameter is :py:class:`~modin.experimental.xgboost.DMatrix` and (2) a new parameter `num_actors`.

.. autofunction:: modin.experimental.xgboost.train

Internal execution flow on Ray backend
''''''''''''''''''''''''''''''''''''''

Internal functions :py:func:`~modin.experimental.xgboost.xgboost_ray._train` and
:py:func:`~modin.experimental.xgboost.xgboost_ray._predict` work similar to xgboost. Approximate execution flow of
internal implementation is the following:

1. The data is passed to :py:func:`~modin.experimental.xgboost.xgboost_ray._train`/:py:func:`~modin.experimental.xgboost.xgboost_ray._predict`
   function as a :py:class:`~modin.experimental.xgboost.DMatrix` object. Using an iterator of
   :py:class:`~modin.experimental.xgboost.DMatrix`, lists of ``ray.ObjectRef`` with row partitions of Modin DataFrame are exctracted. Example:

   .. code-block:: python

     # Extract lists of row partitions from dtrain (DMatrix object)
     X_row_parts, y_row_parts = dtrain
   ..

2. On this step, the parameter `num_actors` is processed. The internal function :py:func:`~modin.experimental.xgboost.xgboost_ray._get_num_actors`
   examines the value provided by the user and checks if it fits in the set of expected values
   (int, "default_train", "default_predict").

   * int - `num_actors` won't be changed. This value will be used.
   * "default_train" - `num_actors` will be computed using condition that 1 actor should use maximum 2 CPUs.
     This condition was chosen for using maximum parallel workers with multithreaded XGBoost training (2 threads
     per worker will be used in this case).
   * "default_predict" - `num_actors` will be computed using condition that 1 actor should use maximum 8 CPUs.
     This condition was chosen to combine parallelization techniques: parallel actors and parallel threads.

.. note:: `num_actors` parameter is made available for public functions :py:func:`~modin.experimental.xgboost.train` and ``predict``
  method of :py:class:`~modin.experimental.xgboost.Booster` class to allow fine-tuning for obtaining the best
  performance in specific use cases.

3. ``ray.util.placement_group`` is created to reserve all available Ray resources. After that
   :py:class:`~modin.experimental.xgboost.xgboost_ray.ModinXGBoostActor` objects are created using resources of the
   previously created placement group.

4. Data (`dtrain` for :py:func:`~modin.experimental.xgboost.xgboost_ray._train`, `data` for
   :py:func:`~modin.experimental.xgboost.xgboost_ray._predict`) is split between actors evenly. The internal function
   :py:func:`~modin.experimental.xgboost.xgboost_ray._split_data_across_actors` runs assigning row partitions to actors
   using internal function :py:func:`~modin.experimental.xgboost.xgboost_ray._assign_row_partitions_to_actors`.
   This function creates a dictionary in the form: `{actor_rank: ([part_i0, part_i3, ..], [0, 3, ..]), ..}` for training,
   `{actor_rank: [part_i0, part_i1, ..], ..}` for prediction.

.. note:: :py:func:`~modin.experimental.xgboost.xgboost_ray._assign_row_partitions_to_actors` takes into account IP
  addresses of row partitions of `dtrain` data to minimize excess data transfer.

1. For each :py:class:`~modin.experimental.xgboost.xgboost_ray.ModinXGBoostActor` the object methods ``set_train_data`` or ``set_predict_data`` are
   called remotely. Those methods run by loading row partitions in actor according to the dictionary with partitions
   distribution from previous step. When data is passed to the actor, the row partitions are automatically materialized
   (``ray.ObjectRef`` -> ``pandas.DataFrame``).

2. Methods ``train`` or ``predict`` of :py:class:`~modin.experimental.xgboost.xgboost_ray.ModinXGBoostActor` class object are called remotely.

   * ``train``: method runs XGBoost training on local data of actor, connects to ``Rabit Tracker`` for sharing
     training state between actors and returns dictionary with `booster` and `evaluation results`.
   * ``predict``: method runs XGBoost prediction on local data of actor and returns IP address of actor and partial
     prediction (``pandas.DataFrame``).

3. On the final stage results from actors are returned.

   * ``train``: `booster` and `evals_result` is returned using ``ray.get`` function from remote actor. Placement
     group which was created on the step 3 is removed to free resources. :py:class:`~modin.experimental.xgboost.Booster`
     object is created and returned to user. 
   * ``predict``: using ``ray.wait`` function we wait until all actors finish computing local predictions. Placement group
     which was created on the step 3 is removed to free resources. ``modin.pandas.DataFrame`` is created from
     ``ray.ObjectRef`` objects which is returned from actors. Modin DataFrame is returned to user.


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
