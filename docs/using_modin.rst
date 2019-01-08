Using Modin
===========

Modin is an early stage DataFrame library that wraps pandas and transparently
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

Using Modin on a Cluster
------------------------

Modin can be run on a cluster, but the setup process is quite complex. We are working on
a solution to make Modin run on a cluster with a simple setup. More on this coming soon!

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

If you are working with very large files or would like to exceed your memory, you may
change the primary location of the DataFrame. If you would like to exceed memory, you
can use your disk as backup for the memory. This API is experimental in the context of
Modin. Please let us know what you think!

Instead of limiting the size of your DataFrame to the amount of memory you have, you can
back your memory with disk:

.. code-block:: python

   import ray
   num_bytes = 2**40 # Make sure you have disk space to do this!
   ray.init(plasma_directory="/tmp", object_store_memory=num_bytes)
   import modin.pandas as pd

Setting ``plasma_directory="/tmp"`` uses your disk for storing the DataFrame and setting
``object_store_memory`` sets the maximum size of the plasma store.

Note: This will impact performance for most operations. This should be used when you are
trying to use very large datasets.

**Warning: Make sure you have enough space in your disk for however many bytes you**
**request for your DataFrame**

Reducing or limiting the resources Modin can use
""""""""""""""""""""""""""""""""""""""""""""""""

By default, Modin will use all of the resources available on your machine. It is
possible, however, to limit the amount of resources Modin uses to free resources for
another task or user. Here is how you would limit the number of CPUs Modin used:

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

.. _`installation page`: http://modin.readthedocs.io/en/latest/installation.html
.. _`currently supported methods`: http://modin.readthedocs.io/en/latest/pandas_supported.html
.. _`open an issue`: http://github.com/modin-project/modin/issues
.. _Ray's documentation: https://ray.readthedocs.io/en/latest/api.html
.. _`blog post`: https://rise.cs.berkeley.edu/blog/pandas-on-ray-early-lessons/ 
.. _`Jupyter Notebook`: http://gist.github.com/devin-petersohn/f424d9fb5579a96507c709a36d487f24#file-pandas_on_ray_blog_post_0-ipynb
