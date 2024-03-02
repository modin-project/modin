Partition API in Modin
======================

When you are working with a :py:class:`~modin.pandas.dataframe.DataFrame`, you can unwrap its remote partitions
to get the raw futures objects compatible with the execution engine (e.g. ``ray.ObjectRef`` for Ray).
In addition to unwrapping of the remote partitions we also provide an API to construct a ``modin.pandas.DataFrame``
from raw futures objects.

Partition IPs
-------------
For finer grained placement control, Modin also provides an API to get the IP addresses of the nodes that hold each partition.
You can pass the partitions having needed IPs to your function. It can help with minimizing of data movement between nodes.

Partition API implementations
-----------------------------
By default, a :py:class:`~modin.pandas.dataframe.DataFrame` stores underlying partitions as ``pandas.DataFrame`` objects.
You can find the specific implementation of Modin's Partition Interface in :doc:`pandas Partition API </flow/modin/distributed/dataframe/pandas>`.

.. toctree::
  :hidden:

  /flow/modin/distributed/dataframe/pandas

Ray engine
----------
However, it is worth noting that for Modin on ``Ray`` engine with ``pandas`` in-memory format IPs of the remote partitions may not match
actual locations if the partitions are lower than 100 kB. Ray saves such objects (<= 100 kB, by default) in in-process store
of the calling process (please, refer to `Ray documentation`_ for more information). We can't get IPs for such objects while maintaining good performance.
So, you should keep in mind this for unwrapping of the remote partitions with their IPs. Several options are provided to handle the case in
``How to handle Ray objects that are lower 100 kB`` section.

Dask engine
-----------
There is no mentioned above issue for Modin on ``Dask`` engine with ``pandas`` in-memory format because ``Dask`` saves any objects
in the worker process that processes a function (please, refer to `Dask documentation`_ for more information).

Unidist engine
--------------
Currently, Modin only supports MPI through unidist. There is no mentioned above issue for
Modin on ``Unidist`` engine using ``MPI`` backend with ``pandas`` in-memory format
because ``Unidist`` saves any objects in the MPI worker process that processes a function
(please, refer to `Unidist documentation`_ for more information).

How to handle Ray objects that are lower than 100 kB
----------------------------------------------------

* If you are sure that each of the remote partitions being unwrapped is higher than 100 kB, you can just import Modin or perform ``ray.init()`` manually.

* If you don't know partition sizes you can pass the option ``_system_config={"max_direct_call_object_size": <nbytes>,}``, where ``nbytes`` is threshold for objects that will be stored in in-process store, to ``ray.init()``.

* You can also start Ray as follows: ``ray start --head --system-config='{"max_direct_call_object_size":<nbytes>}'``.

Note that when specifying the threshold the performance of some Modin operations may change.

.. _`Ray documentation`: https://docs.ray.io/en/master/index.html#
.. _`Dask documentation`: https://distributed.dask.org/en/latest/index.html
.. _`Unidist documentation`: https://unidist.readthedocs.io/en/latest/index.html
