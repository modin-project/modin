Partition IPs in Modin (experimental)
======================================

If you are working with Modin DataFrame and would like to unwrap its remote partitions
for your needs  (pass them to another function that will be processed on a concrete node of the cluster,
for instance), you can use IPs of the remote partitions. In that case you can pass the partitions
having needed IPs to your function. It can help with minimazing of data movement between nodes. However,
it worth noticing that for Modin on ``Ray`` engine with ``pandas`` backend IPs of the remote partitions may not match
actual locations if the partitions are lower 100 kB. Ray saves such objects (<= 100 kB, by default) in in-process store
of the calling process. We can't get IPs for such objects with saving good performance. So, you should keep in mind this
when unwrapping of the remote partitions with their IPs. Several options are provided to handle the case in
``How to handle objects that are lower 100 kB`` section. Wherein, there is no such issue for Modin on ``Dask`` engine
with ``pandas`` backend because ``Dask`` saves any objects in the worker process that processes a function.
Please let us know what you think!

Install Modin Partition IPs
----------------------------

Modin now comes with all the dependencies for partitions IPs functionality by default! See
the `installation page`_ for more information on installing Modin.

Starting Modin with Partition IPs enabled
------------------------------------------

Partition IPs is detected from an environment variable set in bash.

.. code-block:: bash

   export MODIN_ENABLE_PARTITIONS_API=true

How to handle objects that are lower 100 kB
-------------------------------------------

* If you are sure that each of the remote partitions unwrapped is higher 100 kB, you can just import Modin or perform ``ray.init()`` manually.

* If you don't know partitions size you can pass the option ``_system_config={"max_direct_call_object_size": <nbytes>,}``, where ``nbytes`` is threshold for objects that will be stored in in-process store, to ``ray.init()`` or export the following environment variable:

.. code-block:: bash

   export MODIN_ON_RAY_PARTITION_THRESHOLD=<nbytes>

When specifying ``nbytes`` is equal to 0, all the objects will be saved to shared-memory object store (plasma).

* You can also start Ray as follows: ``ray start --head --system-config='{"max_direct_call_object_size":<nbytes>}'``.

Note that when specifying the threshold the performance of some Modin operations may change.

Running an example with Partition IPs
--------------------------------------

Before you run this, please make sure you follow the instructions listed above.

.. code-block:: python

  import modin.pandas as pd
  from modin.api import unwrap_partitions, create_df_from_partitions
  df = pd.read_csv("/path/to/your/file")
  partitions = unwrap_partitions(df, axis=0, bind_ip=True)
  print(partitions)
  # Also, you can create Modin DataFrame from remote partitions including their IPs
  new_df = create_df_from_partitions(partitions, 0)
  print(new_df)

.. _`installation page`: installation.rst
