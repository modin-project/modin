=====================
Using Modin Locally
=====================

.. note:: 
  | *Estimated Reading Time: 5 minutes*

..   | You can follow along this tutorial in a Jupyter notebook `here <hhttps://github.com/modin-project/modin/tree/master/examples/tutorial/tutorial_notebooks/introduction/quickstart.  ipynb>`. 

In our quickstart example, we have already seen how you can achieve considerable speedup from Modin, even on a single machine. 
Users do not need to know how many cores their system has, nor do
they need to specify how to distribute the data. In fact, users can **continue using their
existing pandas code** while experiencing a considerable speedup from Modin, even on
a single machine. 

To use Modin on a single machine, only a modification of the import statement is needed. Once you've changed your import statement, you're ready to use Modin
just like you would pandas, since the API is identical to pandas.

.. code-block:: python

  # import pandas as pd
  import modin.pandas as pd

**That's it. You're ready to use Modin on your previous pandas workflows!** 

Optional Configurations
-------------------------

When using Modin locally on a single machine or laptop (without a cluster), Modin will automatically create and manage a local Dask or Ray cluster for the executing your code. So when you run an operation for the first time with Modin, you will see a message like this, indicating that a Modin has automatically initialized a cluster for you:

.. code-block:: python

  df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})

.. code-block:: text

   UserWarning: Ray execution environment not yet initialized. Initializing...
   To remove this warning, run the following python code before doing dataframe operations:

    import ray
    ray.init()

 If you prefer to use Dask over Ray as your execution backend, you can use the following code to modify the default configuration:

.. code-block:: python

   import modin
   modin.config.Engine.put("Dask")

.. code-block:: python

   df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})


.. code-block:: text

   UserWarning: Dask execution environment not yet initialized. Initializing...
   To remove this warning, run the following python code before doing dataframe operations:

      from distributed import Client

      client = Client()

Finally, if you already have an Ray or Dask engine initialized, Modin will automatically attach to whichever engine is available. If you are interested in using Modin with OmniSci engine, please refer to :doc:`these instructions </developer/using_omnisci>`. For additional information on other settings you can configure, see :doc:`this page </flow/modin/config>` for more details.

Advanced: Configuring the resources Modin uses
-----------------------------------------------

Modin automatically check the number of CPUs available on your machine and sets the number of partitions to be equal to the number of CPUs. You can verify this by running the following code:

.. code-block:: python

   import modin
   print(modin.config.NPartitions.get()) #prints 16 on a laptop with 16 physical cores

Modin fully utilizes the resources on your machine. To read more about how this works, see :doc:`this page</getting_started/why_modin/pandas/>` for more details.

Since Modin will use all of the resources available on your machine by default, at times, it is possible that you may like to limit the amount of resources Modin uses to free resources for
another task or user. Here is how you would limit the number of CPUs Modin used in
your bash environment variables:

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
   Make sure to update the ``MODIN_CPUS`` configuration and initialize your preferred engine before you start working with the first operation using Modin! Otherwise Modin will opt for the default setting.