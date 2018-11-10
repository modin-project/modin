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

  NotImplementedError: To contribute to Pandas on Ray, please visit github.com/modin-project/modin.

We have compiled a list of `currently supported methods`_.

If you would like to request a particular method be implemented, feel free to `open an
issue`_. Before you open an issue please make sure that someone else has not already
requested that functionality.

Using Modin on a Cluster
------------------------

Modin can be run on a cluster, but the setup process is quite complex. We are working on
a solution to make Modin run on a cluster with a simple setup. More on this coming soon!

Examples
--------
You can find an example on our recent `blog post`_ or on the `Jupyter Notebook`_ that we
used to create the blog post.

.. _`installation page`: http://modin.readthedocs.io/en/latest/installation.html
.. _`currently supported methods`: http://modin.readthedocs.io/en/latest/pandas_supported.html
.. _`open an issue`: http://github.com/modin-project/modin/issues
.. _`blog post`: http://rise.cs.berkeley.edu/blog/pandas-on-ray-lessons-learned
.. _`Jupyter Notebook`: http://gist.github.com/devin-petersohn/f424d9fb5579a96507c709a36d487f24#file-pandas_on_ray_blog_post_0-ipynb
