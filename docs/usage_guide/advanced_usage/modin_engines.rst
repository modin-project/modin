Modin engines
=============

As a rule, you don't have to worry about initialization of an execution engine as
Modin itself automatically initializes one when performing the first operation.
Also, Modin has a broad range of :doc:`configuration settings </flow/modin/config>`, which
you can use to configure an execution engine. If there is a reason to initialize an execution engine
on your own and you are sure what to do, Modin will automatically attach to whichever engine is available.
Below, you can find some examples on how to initialize a specific execution engine on your own.

Ray
---

You can initialize Ray engine with a specific number of CPUs (worker processes) to perform computation.

.. code-block:: python

  import ray
  import modin.config as modin_cfg

  ray.init(num_cpus=<N>)
  modin_cfg.Engine.put("ray") # Modin will use Ray engine
  modin_cfg.CpuCount.put(<N>)

To get more details on all possible parameters for initialization refer to `Ray documentation`_.

Dask
----

You can initialize Dask engine with a specific number of worker processes and threads per worker to perform computation.

.. code-block:: python

  from distributed import Client
  import modin.config as modin_cfg

  client = Client(n_workers=<N1>, threads_per_worker=<N2>)
  modin_cfg.Engine.put("dask") # # Modin will use Dask engine
  modin_cfg.CpuCount.put(<N1>)

To get more details on all possible parameters for initialization refer to `Dask Distributed documentation`_.

MPI through unidist
-------------------

You can initialize MPI through unidist engine with a specific number of CPUs (worker processes) to perform computation.

.. code-block:: python

  import unidist
  import unidist.config as unidist_cfg
  import modin.config as modin_cfg

  unidist_cfg.Backend.put("mpi")
  unidist_cfg.CpuCount.put(<N>)
  unidist.init()

  modin_cfg.Engine.put("unidist") # # Modin will use MPI through unidist engine
  modin_cfg.CpuCount.put(<N>)

To get more details on all possible parameters for initialization refer to `unidist documentation`_.

.. _`Ray documentation`: https://docs.ray.io/en/latest
.. _Dask Distributed documentation: https://distributed.dask.org/en/latest
.. _`unidist documentation`: https://unidist.readthedocs.io/en/latest
