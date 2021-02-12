Pandas Partition API in Modin
=============================

If you are working with a Modin Dataframe, you can unwrap its remote partitions
to get the raw future objects compatible with the execution engine (e.g. ``ray.ObjectRef`` for Ray).
You can use this API to get the IPs of the nodes that hold these objects as well. In that case you can pass the partitions
having needed IPs to your function. It can help with minimazing of data movement between nodes. In addition to
unwrapping of the remote partitions we also provide API to construct Modin DataFrame from them.

Ray engine
----------
However, it is worth noting that for Modin on ``Ray`` engine with ``pandas`` backend IPs of the remote partitions may not match
actual locations if the partitions are lower than 100 kB. Ray saves such objects (<= 100 kB, by default) in in-process store
of the calling process (please, refer to `Ray documentation`_ for more information). We can't get IPs for such objects while maintaining good performance.
So, you should keep in mind this for unwrapping of the remote partitions with their IPs. Several options are provided to handle the case in
``How to handle Ray objects that are lower 100 kB`` section.

Dask engine
-----------
There is no mentioned above issue for Modin on ``Dask`` engine with ``pandas`` backend because ``Dask`` saves any objects
in the worker process that processes a function (please, refer to `Dask documentation`_ for more information).

Install Modin Pandas Partition API
----------------------------------

Modin now comes with all the dependencies for pandas partition API functionality by default! See
the :doc:`installation page </installation>` for more information on installing Modin.

How to handle Ray objects that are lower than 100 kB
----------------------------------------------------

* If you are sure that each of the remote partitions being unwrapped is higher than 100 kB, you can just import Modin or perform ``ray.init()`` manually.

* If you don't know partition sizes you can pass the option ``_system_config={"max_direct_call_object_size": <nbytes>,}``, where ``nbytes`` is threshold for objects that will be stored in in-process store, to ``ray.init()`` or export the following environment variable:

.. code-block:: bash

   export MODIN_ON_RAY_PARTITION_THRESHOLD=<nbytes>

When specifying ``nbytes`` equal to 0, all the objects will be saved to shared-memory object store (plasma).

* You can also start Ray as follows: ``ray start --head --system-config='{"max_direct_call_object_size":<nbytes>}'``.

Note that when specifying the threshold the performance of some Modin operations may change.

API
---

It is currently supported the following API:

.. automodule:: modin.distributed.dataframe.pandas
  :noindex:
  :members: unwrap_partitions

.. automodule:: modin.distributed.dataframe.pandas
  :noindex:
  :members: from_partitions

Running an example with pandas partition API
--------------------------------------------

Before you run this, please make sure you follow the instructions listed above.

.. code-block:: python

  import modin.pandas as pd
  from modin.distributed.dataframe.pandas import unwrap_partitions, from_partitions
  import numpy as np
  data = np.random.randint(0, 100, size=(2 ** 10, 2 ** 8))
  df = pd.DataFrame(data)
  partitions = unwrap_partitions(df, axis=0, get_ip=True)
  print(partitions)
  # Also, you can create Modin DataFrame from remote partitions including their IPs
  new_df = from_partitions(partitions, axis=0)
  print(new_df)


.. _`Ray documentation`: https://docs.ray.io/en/master/index.html#
.. _`Dask documentation`: https://distributed.dask.org/en/latest/index.html
