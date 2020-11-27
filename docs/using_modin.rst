Using Modin
===========

Modin is an early stage `DataFrame`_ library that wraps `pandas`_ and transparently
distributes the data and computation, accelerating your pandas workflows with one line
of code change. The user does not need to know how many cores their system has, nor do
they need to specify how to distribute the data. In fact, users can continue using their
previous pandas notebooks while experiencing a considerable speedup from Modin, even on
a single machine. Only a modification of the import statement is needed, as we
demonstrate below. Once you’ve changed your import statement, you’re ready to use Modin
just like you would pandas, since the API is identical to pandas.

.. code-block:: python

  # import pandas as pd
  import modin.pandas as pd

Currently, we have part of the pandas API implemented and are working toward full
functional parity with pandas.

Using Modin on a Single Node
----------------------------

**In local (without a cluster) modin will create and manage a local (dask or ray) cluster for the execution**

In order to use the most up-to-date version of Modin, please follow the instructions on
the `installation page`_

Once you import the library, you should see something similar to the following output:

.. code-block:: text

  >>> import modin.pandas as pd

  Waiting for redis server at 127.0.0.1:14618 to respond...
  Waiting for redis server at 127.0.0.1:31410 to respond...
  Starting local scheduler with the following resources: {'CPU': 4, 'GPU': 0}.

  ======================================================================
  View the web UI at http://localhost:8889/notebooks/ray_ui36796.ipynb?token=ac25867d62c4ae87941bc5a0ecd5f517dbf80bd8e9b04218
  ======================================================================

Once you have executed  ``import modin.pandas as pd``, you're ready to begin
running your pandas pipeline as you were before.

APIs Supported
--------------

Please note, the API is not yet complete. For some methods, you may see the following:

.. code-block:: text

  NotImplementedError: To contribute to Modin, please visit github.com/modin-project/modin.

We have compiled a list of `currently supported methods`_.

If you would like to request a particular method be implemented, feel free to `open an
issue`_. Before you open an issue please make sure that someone else has not already
requested that functionality.

Using Modin on a Cluster (experimental)
---------------------------------------

Modin is able to utilize Ray's built-in autoscaled cluster. However, this usage
is still under heavy development. To launch a Ray autoscaled cluster using
Amazon Web Service (AWS), you can use the file `examples/cluster/aws_example.yaml`
as the config file when launching an autoscaled Ray cluster. For the commands,
refer to the `autoscaler documentation`_.

We will provide a sample config file for private servers and other cloud service
providers as we continue to develop and improve Modin's cluster support.

Advanced usage (experimental)
-----------------------------

In some cases, it may be useful to customize your Ray environment. Below, we have listed
a few ways you can solve common problems in data management with Modin by customizing
your Ray environment. It is possible to use any of Ray's initialization parameters,
which are all found in `Ray's documentation`_.

.. code-block:: python

   import ray
   ray.init()
   import modin.pandas as pd

Modin will automatically connect to the Ray instance that is already running. This way,
you can customize your Ray environment for use in Modin!

Exceeding memory (Out of core pandas)
"""""""""""""""""""""""""""""""""""""

Modin experimentally supports out of core operations. See more on the `Out of Core`_
documentation page.

Reducing or limiting the resources Modin can use
""""""""""""""""""""""""""""""""""""""""""""""""

By default, Modin will use all of the resources available on your machine. It is
possible, however, to limit the amount of resources Modin uses to free resources for
another task or user. Here is how you would limit the number of CPUs Modin used in
your bash environment variables:

.. code-block:: bash

   export MODIN_CPUS=4


You can also specify this in your python script with ``os.environ``. **Make sure
you update the CPUS before you import Modin!**:

.. code-block:: python

   import os
   os.environ["MODIN_CPUS"] = "4"
   import modin.pandas as pd

If you're using a specific engine and want more control over the environment Modin
uses, you can start Ray or Dask in your environment and Modin will connect to it.
**Make sure you start the environment before you import Modin!**

.. code-block:: python

   import ray
   ray.init(num_cpus=4)
   import modin.pandas as pd

Specifying ``num_cpus`` limits the number of processors that Modin uses. You may also
specify more processors than you have available on your machine, however this will not
improve the performance (and might end up hurting the performance of the system).

Examples
--------
You can find an example on our recent `blog post`_ or on the `Jupyter Notebook`_ that we
used to create the blog post.

.. _`DataFrame`: https://pandas.pydata.org/pandas-docs/version/0.23.4/generated/pandas.DataFrame.html
.. _`pandas`: https://pandas.pydata.org/pandas-docs/stable/
.. _`installation page`: https://modin.readthedocs.io/en/latest/installation.html
.. _`currently supported methods`: https://modin.readthedocs.io/en/latest/pandas_supported.html
.. _`open an issue`: https://github.com/modin-project/modin/issues
.. _`autoscaler documentation`: https://ray.readthedocs.io/en/latest/autoscaling.html
.. _`Ray's documentation`: https://ray.readthedocs.io/en/latest/api.html
.. _`blog post`: https://rise.cs.berkeley.edu/blog/pandas-on-ray-early-lessons/
.. _`Jupyter Notebook`: https://gist.github.com/devin-petersohn/f424d9fb5579a96507c709a36d487f24#file-pandas_on_ray_blog_post_0-ipynb
.. _`Out of Core`: out_of_core.html
