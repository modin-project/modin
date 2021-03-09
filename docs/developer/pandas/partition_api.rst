Partition API in Modin
======================

When you are working with a Modin Dataframe, you can unwrap its remote partitions
to get the raw futures objects compatible with the execution engine (e.g. ``ray.ObjectRef`` for Ray).
In addition to unwrapping of the remote partitions we also provide an API to construct a ``modin.pandas.DataFrame``
from raw futures objects.

Partition IPs
-------------
For finer grained placement control, Modin also provides an API to get the IP addresses of the nodes that hold each partition.
You can pass the partitions having needed IPs to your function. It can help with minimazing of data movement between nodes.

unwrap_partitions
-----------------

.. automodule:: modin.distributed.dataframe.pandas
  :noindex:
  :members: unwrap_partitions

from_partitions
---------------

.. automodule:: modin.distributed.dataframe.pandas
  :noindex:
  :members: from_partitions

Example
-------

.. code-block:: python

  import modin.pandas as pd
  from modin.distributed.dataframe.pandas import unwrap_partitions, from_partitions
  import numpy as np
  data = np.random.randint(0, 100, size=(2 ** 10, 2 ** 8))
  df = pd.DataFrame(data)
  partitions = unwrap_partitions(df, axis=0, get_ip=True)
  print(partitions)
  new_df = from_partitions(partitions, axis=0)
  print(new_df)

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

How to handle Ray objects that are lower than 100 kB
----------------------------------------------------

* If you are sure that each of the remote partitions being unwrapped is higher than 100 kB, you can just import Modin or perform ``ray.init()`` manually.

* If you don't know partition sizes you can pass the option ``_system_config={"max_direct_call_object_size": <nbytes>,}``, where ``nbytes`` is threshold for objects that will be stored in in-process store, to ``ray.init()``.

* You can also start Ray as follows: ``ray start --head --system-config='{"max_direct_call_object_size":<nbytes>}'``.

Note that when specifying the threshold the performance of some Modin operations may change.

.. _`Ray documentation`: https://docs.ray.io/en/master/index.html#
.. _`Dask documentation`: https://distributed.dask.org/en/latest/index.html
