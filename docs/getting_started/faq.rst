Frequently Asked Questions (FAQs)
=================================

Below, you will find answers to the most commonly asked questions about
Modin. If you still cannot find the answer you are looking for, please post your
question on the #support channel on our Slack_ community or open a Github issue_.

What’s wrong with pandas and why should I use Modin?
""""""""""""""""""""""""""""""""""""""""""""""""""""

While pandas works extremely well on small datasets, as soon as you start working with
medium to large datasets that are more than a few GBs, pandas can become painfully
slow or run out of memory. This is because pandas is single-threaded. In other words,
you can only process your data with one core at a time. This approach does not scale to
larger data sets and adding more hardware does not lead to more performance gain.

The :py:class:`~modin.pandas.dataframe.DataFrame` is a highly
scalable, parallel DataFrame. Modin transparently distributes the data and computation so
that you can continue using the same pandas API while being able to work with more data faster.
Modin lets you use all the CPU cores on your machine, and because it is lightweight, it
often has less memory overhead than pandas. See this :doc:`page </getting_started/pandas>` to
learn more about how Modin is different from pandas.

Why not just improve pandas?
""""""""""""""""""""""""""""

pandas is a massive community and well established codebase. Many of the issues
we have identified and resolved with pandas are fundamental to its current
implementation. While we would be happy to donate parts of Modin that
make sense in pandas, many of these components would require significant (or
total) redesign of the pandas architecture. Modin's architecture goes beyond
pandas, which is why the pandas API is just a thin layer at the user level. To learn
more about Modin's architecture, see the :doc:`architecture </development/architecture>` documentation.

How much faster can I go with Modin compared to pandas?
"""""""""""""""""""""""""""""""""""""""""""""""""""""""

Modin is designed to scale with the amount of hardware available.
Even in a traditionally serial task like ``read_csv``, we see large gains by efficiently
distributing the work across your entire machine. Because it is so light-weight,
Modin provides speed-ups of up to 4x on a laptop with 4 physical cores. This speedup scales
efficiently to larger machines with more cores. We have several published papers_ that
include performance results and comparisons against pandas.

How much more data would I be able to process with Modin?
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""

Often data scientists have to use different tools for operating on datasets of different sizes.
This is not only because processing large dataframes is slow, but also pandas does not support working
with dataframes that don't fit into the available memory. As a result, pandas workflows that work well
for prototyping on a few MBs of data do not scale to tens or hundreds of GBs (depending on the size
of your machine). Modin supports operating on data that does not fit in memory, so that you can comfortably
work with hundreds of GBs without worrying about substantial slowdown or memory errors. For more information,
see :doc:`out-of-memory support <getting_started/out_of_core.rst>` for Modin.

How does Modin work under the hood?
"""""""""""""""""""""""""""""""""""

Modin is logically separated into different layers that represent the hierarchy of a
typical Database Management System. User queries which perform data transformation,
data ingress or data egress pass through the Modin Query Compiler which translates
queries from the top-level pandas API Layer that users interact with to the Modin Core
Dataframe layer.
The Modin Core DataFrame is our efficient DataFrame implementation that utilizes a partitioning schema
which allows for distributing tasks and queries. From here, the Modin DataFrame works with engines like
Ray or Dask to execute computation, and then return the results to the user.

For more details, take a look at our system :doc:`architecture </development/architecture>`.

If I’m only using my laptop, can I still get the benefits of Modin?
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

Absolutely! Unlike other parallel DataFrame systems, Modin is an extremely
light-weight, robust DataFrame. Because it is so light-weight, Modin provides
speed-ups of up to 4x on a laptop with 4 physical cores
and allows you to work on data that doesn't fit in your laptop's RAM.

How do I use Jupyter or Colab notebooks with Modin?
"""""""""""""""""""""""""""""""""""""""""""""""""""

You can take a look at this Google Colab installation guide_ and
this notebook tutorial_. Once Modin is installed, simply replace your pandas
import with Modin import:

.. code-block:: python

    # import pandas as pd
    import modin.pandas as pd

Which execution engine (Ray or Dask) should I use for Modin?
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

Whichever one you want! Modin supports Ray_ and Dask_ execution engines to provide an effortless way
to speed up your pandas workflows. The best thing is that you don't need to know
anything about Ray and Dask in order to use Modin and Modin will automatically
detect which engine you have
installed and use that for scheduling computation. If you don't have a preference, we recommend
starting with Modin's default Ray engine. If you want to use a specific
compute engine, you can set the environment variable ``MODIN_ENGINE`` and
Modin will do computation with that engine:

.. code-block:: bash

    pip install "modin[ray]" # Install Modin dependencies and Ray to run on Ray
    export MODIN_ENGINE=ray  # Modin will use Ray

    pip install "modin[dask]" # Install Modin dependencies and Dask to run on Dask
    export MODIN_ENGINE=dask  # Modin will use Dask

We also have an experimental OmniSciDB-based engine of Modin you can read about :doc:`here </development/using_omnisci>`.
We plan to support more execution engines in future. If you have a specific request,
please post on the #feature-requests channel on our Slack_ community.

How can I contribute to Modin?
""""""""""""""""""""""""""""""

**Modin is currently under active development. Requests and contributions are welcome!**

If you are interested in contributing please check out the :doc:`Getting Started</getting_started/index>`
guide then refer to the :doc:`Development Documentation</development/index>` section,
where you can find system architecture, internal implementation details, and other useful information.
Also check out the `Github`_ to view open issues and make contributions.

.. _issue: https://github.com/modin-project/modin/issues
.. _Slack: https://modin.org/slack.html
.. _Github: https://github.com/modin-project/modin
.. _Ray: https://github.com/ray-project/ray/
.. _Dask: https://dask.org/
.. _papers: https://arxiv.org/abs/2001.00888
.. _guide: https://modin.readthedocs.io/en/stable/installation.html?#installing-on-google-colab
.. _tutorial: https://github.com/modin-project/modin/tree/master/examples/tutorial
