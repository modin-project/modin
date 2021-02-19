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

map_partitions_to_ips
---------------------

.. automodule:: modin.distributed.dataframe.pandas
  :noindex:
  :members: map_partitions_to_ips

from_partitions
---------------

.. automodule:: modin.distributed.dataframe.pandas
  :noindex:
  :members: from_partitions

Example
-------

.. code-block:: python

  import modin.pandas as pd
  from modin.distributed.dataframe.pandas import (
    unwrap_partitions,
    map_partitions_to_ips,
    from_partitions,
  )
  import numpy as np
  data = np.random.randint(0, 100, size=(2 ** 10, 2 ** 8))
  df = pd.DataFrame(data)
  partitions = unwrap_partitions(df, axis=0, get_ip=True)
  print(partitions)
  mapped_partitions = map_partitions_to_ips(partitions, axis=0)
  print(mapped_partitions)
  new_df = from_partitions(partitions, axis=0)
  print(new_df)
