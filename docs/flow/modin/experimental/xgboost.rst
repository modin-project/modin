Modin XGBoost module description
""""""""""""""""""""""""""""""""
High-level Module Overview
''''''''''''''''''''''''''

This module holds classes, public interface functions and internal functions for making distributed 
xgboost in Modin.

Public classes :ref:`Booster <Booster>`, :ref:`DMatrix <DMatrix>` and function  :ref:`train <train>` provide
to user familiar interfaces of xgboost. That functionality is placed in ``modin.experimental.xgboost.xgboost`` module.

Internal module ``modin.experimental.xgboost.xgboost.xgboost_ray`` holds implementation of Modin XGBoost
for Ray backend. This module mainly consist of Ray actor-class :ref:`ModinXGBoostActor <Actor>`,
function to distribution Modin's partitions between actors :ref:`_assign_row_partitions_to_actors <assign_parts>`,
internal :ref:`_train <internal_train>`/:ref:`_predict <internal_predict>` functions used from the public interfaces 
and additional util functions for computing cluster resources, actor creations etc.

Public interfaces
'''''''''''''''''

Class :ref:`DMatrix <DMatrix>` inherits original class ``xgboost.DMatrix`` and overrides constructor which supports
only `data` and `label` parameters now. Both of the parameters must be ``modin.pandas.DataFrame``,
those dataframes will be unwrapped to lists of delayed objects of Modin's row partitions using function
``modin.distributed.dataframe.pandas.unwrap_partitions``.

.. _DMatrix:
.. autoclass:: modin.experimental.xgboost.DMatrix
  :members:

Class :ref:`Booster<Booster>` inherits original class ``xgboost.Booster`` and overrides method ``predict``.
The one of the differences from original class interface for ``predict`` method is in changing type of `data`
parameter to :ref:`DMatrix <DMatrix>`, the second one is new parameter `num_actors`,
which specify number of actors to run for prediction.

.. _Booster:
.. autoclass:: modin.experimental.xgboost.Booster
    :noindex:
    :members:

:ref:`train <train>` function (similary to ``predict`` method of :ref:`Booster <Booster>`) has 2 differences
from original ``train`` function - data type of `dtrain` parameter is :ref:`DMatrix <DMatrix>` and new
parameter `num_actors`.

.. _train:
.. autofunction:: modin.experimental.xgboost.train
    :noindex:

Internal execution flow on Ray backend
''''''''''''''''''''''''''''''''''''''

Internal functions :ref:`_train <internal_train>` and :ref:`_predict <internal_predict>` work similary.
Approximate execution flow of internal implementation is the following:

1. The data arrives to :ref:`_train <internal_train>`/:ref:`_predict <internal_predict>` function as the
   :ref:`DMatrix <DMatrix>` object. Using iterator of :ref:`DMatrix <DMatrix>`, lists of ``ray.ObjectRef``
   with row partitions of Modin DataFrame are exctracted. Example:

   .. code-block:: python

     # Extract lists of row partitions from dtrain (DMatrix object)
     X_row_parts, y_row_parts = dtrain
   ..

2. On this step parameter ``num_actors`` is processed. Internal function :ref:`_get_num_actors <get_num_actors>`
   examines value provided by user and checks if it fits in the set of expected values
   (int, "default_train", "default_predict").

   * int - ``num_actors`` won't be changed. This value will be used.
   * "default_train" - ``num_actors`` will be computed using condition that 1 actor should use maximum 2 CPUs.
     This condition was chosen for using maximum parallel workers with multithreaded xgboost training (2 threads
     per worker will be used in this case).
   * "default_predict" - ``num_actors`` will be computed using condition that 1 actor should use maximum 8 CPUs.
     This condition was chosen to combine  parallelization techniques: parallel actors and parallel threads.

**Note**: ``num_actors`` parameter is made available for public functions :ref:`train <train>` and ``predict``
method of :ref:`Booster<Booster>` class to allow fine-tuning for best performance in specific use cases.

3. ``ray.util.placement_group`` is created to reserve all available Ray resources. After that
   :ref:`ModinXGBoostActor <Actor>` objects are created using resources of previously created placement group.

4. Data (`dtrain` for :ref:`_train <internal_train>`, `data` for :ref:`_predict <internal_predict>`) is split
   between actors evenly. Internal function :ref:`_split_data_across_actors <split_data_across_actors>` runs
   assigning row partitions to actors using internal function :ref:`_assign_row_partitions_to_actors <assign_parts>`.
   This function creates dictionary in the form: `{actor_rank: ([part_i0, part_i3, ..], [0, 3, ..]), ..}` for training,
   `{actor_rank: [part_i0, part_i1, ..], ..}` for prediction.

**Note**: :ref:`_assign_row_partitions_to_actors <assign_parts>` takes into account IP addresses of row partitions
of `dtrain` data to minimize excess data transfer.

5. For each :ref:`ModinXGBoostActor <Actor>` object methods ``set_train_data`` or ``set_predict_data`` are
   called remotely. Those methods run loading row partitions in actor according to dictionary with partitions
   distribution from previous step. When data is passed in actor, auto-materialization of row partitions
   (``ray.ObjectRef`` -> ``pandas.DataFrame``) happens.

6. Methods ``train`` or ``predict`` of :ref:`ModinXGBoostActor <Actor>` class object are called remotely.

   * `train`: method runs xgboost training on local data of actor, connects to ``Rabit Tracker`` for sharing
     training state between actors and returns dictionary with `booster` and `evaluation results`.
   * `predict`: method runs xgboost prediction on local data of actor and returns IP address of actor and partial
     prediction (``pandas.DataFrame``).

7. On the final stage results from actors are returned.

   * `train`: `booster` and `evals_result` is returned using ``ray.get`` function from remote actor. Placement
     group which was created on the step 3 is removed to free resources. :ref:`Booster<Booster>` object is
     created and returned to user. 
   * `predict`: using ``ray.wait`` function we wait until all actors finish computing local predictions. Placement group
     which was created on the step 3 is removed to free resources. ``modin.pandas.DataFrame`` is created from
     ``ray.ObjectRef`` objects which is returned from actors. Modin DataFrame is returned to user.


Internal API
''''''''''''
.. _Actor:
.. autoclass:: modin.experimental.xgboost.xgboost_ray.ModinXGBoostActor
  :members:
  :private-members:

.. _assign_parts:
.. autofunction:: modin.experimental.xgboost.xgboost_ray._assign_row_partitions_to_actors

.. _internal_train:
.. autofunction:: modin.experimental.xgboost.xgboost_ray._train

.. _internal_predict:
.. autofunction:: modin.experimental.xgboost.xgboost_ray._predict

.. _get_num_actors:
.. autofunction:: modin.experimental.xgboost.xgboost_ray._get_num_actors

.. _split_data_across_actors:
.. autofunction:: modin.experimental.xgboost.xgboost_ray._split_data_across_actors
