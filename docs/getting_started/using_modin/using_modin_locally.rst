===================
Using Modin Locally
===================

.. note::
  | *Estimated Reading Time: 5 minutes*
  | You can follow along this tutorial in the `Jupyter notebook`_.

In our quickstart example, we have already seen how you can achieve considerable
speedup from Modin, even on a single machine. Users do not need to know how many
cores their system has, nor do they need to specify how to distribute the data. In fact,
users can **continue using their existing pandas code** while experiencing a
considerable speedup from Modin, even on a single machine.

To use Modin on a single machine, only a modification of the import statement is needed.
Once you've changed your import statement, you're ready to use Modin
just like you would pandas, since the API is identical to pandas.

.. code-block:: python

  # import pandas as pd
  import modin.pandas as pd

**That's it. You're ready to use Modin on your previous pandas workflows!**

Advanced: Configuring the resources Modin uses
----------------------------------------------

Modin automatically check the number of CPUs available on your machine and sets the
number of partitions to be equal to the number of CPUs. You can verify this by running
the following code:

.. code-block:: python

   import modin
   print(modin.config.NPartitions.get()) #prints 16 on a laptop with 16 physical cores

Modin fully utilizes the resources on your machine. To read more about how this works,
see :doc:`Why Modin? </getting_started/why_modin/pandas/>` page for more details.

Since Modin will use all of the resources available on your machine by default, at
times, it is possible that you may like to limit the amount of resources Modin uses to
free resources for another task or user. Here is how you would limit the number of CPUs
Modin used in your bash environment variables:

.. code-block:: bash

   export MODIN_CPUS=4


You can also specify this in your python script with ``os.environ``:

.. code-block:: python

   import os
   os.environ["MODIN_CPUS"] = "4"
   import modin.pandas as pd

If you're using a specific engine and want more control over the environment Modin
uses, you can start Ray or Dask in your environment and Modin will connect to it.

.. code-block:: python

   import ray
   ray.init(num_cpus=4)
   import modin.pandas as pd

Specifying ``num_cpus`` limits the number of processors that Modin uses. You may also
specify more processors than you have available on your machine; however this will not
improve the performance (and might end up hurting the performance of the system).

.. note::
   Make sure to update the ``MODIN_CPUS`` configuration and initialize your preferred
   engine before you start working with the first operation using Modin! Otherwise,
   Modin will opt for the default setting.


.. _`Jupyter notebook`: https://github.com/modin-project/modin/tree/main/examples/quickstart.ipynb
