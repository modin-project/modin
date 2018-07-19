Pandas on Ray
=============

Pandas on Ray is an early stage DataFrame library that wraps Pandas and
transparently distributes the data and computation. The user does not need to
know how many cores their system has, nor do they need to specify how to
distribute the data. In fact, users can continue using their previous Pandas
notebooks while experiencing a considerable speedup from Pandas on Ray, even
on a single machine. Only a modification of the import statement is needed, as
we demonstrate below. Once you’ve changed your import statement, you’re ready
to use Pandas on Ray just like you would Pandas.

.. code-block:: python

  # import pandas as pd
  import modin.pandas as pd

Currently, we have part of the Pandas API implemented and are working toward
full functional parity with Pandas.

Using Pandas on Ray on a Single Node
------------------------------------

In order to use the most up-to-date version of Pandas on Ray, please follow
the instructions on the `installation page`_

Once you import the library, you should see something similar to the following
output:

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

Please note, the API is not yet complete. For some methods, you may see the
following:

.. code-block:: text

  NotImplementedError: To contribute to Pandas on Ray, please visit github.com/modin-project/modin.

We have compiled a list of currently supported methods `here`_.

If you would like to request a particular method be implemented, feel free to
`open an issue`_. Before you open an issue please make sure that someone else
has not already requested that functionality.

Using Pandas on Ray on a Cluster
--------------------------------

Currently, you can run Modin on a cluster using a Jupyter notebook interface.

First, create a config file which specifies the nodes in the cluster.
Then, run ``modin notebook --config=/path/to/config.yaml --port=8890`` from the
console in order to configure the cluster for use with Modin. The command will
launch a Jupyter notebook on the head node and expose it to the local machine
at the specified port.

A config file looks like this:

.. code-block:: yaml

  # The execution engine on which Modin runs. Currently only supports ray.
  execution_engine: ray

  # Optional. The default SSH key used to access nodes.
  key: ~/.ssh/key.pem

  # Configuration for the head node. Requires hostname.
  # Can set an optional key to override the global key.
  head_node:
    hostname: ubuntu@127.0.0.1

  # Configuration for other nodes in the cluster. Each node requires a hostname.
  # For each node, can set an optional key to override the global key.
  nodes:
    - hostname: ubuntu@127.0.0.2
    - hostname: ubuntu@127.0.0.2
      key: ~/.ssh/other_key.pem

Examples
--------
You can find an example on our recent `blog post`_ or on the
`Jupyter Notebook`_ that we used to create the blog post.

.. _`installation page`: http://modin.readthedocs.io/en/latest/installation.html
.. _`here`: http://modin.readthedocs.io/en/latest/pandas_supported.html
.. _`open an issue`: http://github.com/modin-project/modin/issues
.. _`blog post`: http://rise.cs.berkeley.edu/blog/pandas-on-ray
.. _`Jupyter Notebook`: http://gist.github.com/devin-petersohn/f424d9fb5579a96507c709a36d487f24#file-pandas_on_ray_blog_post_0-ipynb
