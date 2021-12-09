Frequently Asked Questions (FAQs)
=================================

Below, you will find answers to the most commonly asked questions about
Modin. If you still cannot find the answer you are looking for, please post on 
the #support channel on our Slack_ community or open a Github issue_.

What’s the issue with pandas? Why should I use Modin?
""""""""""""""""""""""""""""""""""""""""""""""""""""""
While pandas work extremely well on small datasets, as soon as you start working with 
medium to large datasets that are more than a few GBs, pandas can become painfully 
slow or run out of memory. This is because pandas is single-threaded, in other words, 
you can only process your data with one core at a time.
any kind. This approach does not scale to larger data sets and adding more hardware does not
lead to more performance gain. 

The ``modin.pandas`` `DataFrame`_ is a highly scalable, parallel DataFrame. Modin
transparently distributes the data and computation so that you can
continue using the same pandas API while being able to work with more data faster. With Modin, 
you are able to use all of the CPU cores on your machine, and because of it's light-weight
nature often results in less memory overhead than pandas. See this 
:doc:`page </getting_started/pandas>` to learn more about how Modin is different from pandas. 

Why not just improve pandas?
""""""""""""""""""""""""""""
Pandas is a massive community and well established codebase. Many of the issues
we have identified and resolved with pandas are fundamental to its current
implementation. While we would be happy to donate parts of Modin that
make sense in pandas, many of these components would require significant (or
total) redesign of the pandas architecture. Modin's architecture goes beyond
pandas, which is why the pandas API is just a thin layer at the user level. To learn
more about Modin's architecture, see the :doc:`architecture </developer/architecture>` documentation.

How much faster can I go with Modin compared to pandas?
""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Modin is designed to scale with the amount of hardware available.
Even in a traditionally serial task like ``read_csv``, we see large gains by efficiently 
distributing the work across your entire machine. Because it is so light-weight, 
Modin provides speed-ups of up to 4x on a laptop with 4 physical cores. This speedup scales
efficiently to larger machines with more cores. We have several published papers that
include performance results and comparisons against pandas.

How much more data would I be able to process with Modin?
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
We have focused heavily on bridging the solutions between DataFrames for small 
data (e.g. pandas) and large data. Often data scientists require different tools 
for doing the same thing on different sizes of data. The DataFrame solutions that 
exist for 1MB do not scale to 1TB+, and the overheads of the solutions for 1TB+ 
are too costly for datasets in the 1KB range. With Modin, because of its light-weight, 
robust, and scalable nature, you get a fast DataFrame at 1MB and 1TB+.

How does Modin work under the hood?
""""""""""""""""""""""""""""""""""""
Modin is logically separated into different layers that represent the hierarchy of a 
typical Database Management System. User queries which perform data transformation, 
data ingress or data egress pass through the Modin query compiler which translates 
queries from the Pandas API Layer and sends them to the Modin Core DataFrame. The Modin
Core DataFrame has an efficient dataframe partitioning schema which allows for extreme
parallelization. From here, the Modin DataFrame works with task parallel frameworks like
Ray or Dask to execute computation, and then return the results to the user.

For more details, take a look at our system :doc:`architecture </developer/architecture>`. 

If I’m only using my laptop, can I still get the benefits of Modin?
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Absolutely! Unlike other parallel DataFrame systems, Modin is an extremely 
light-weight, robust DataFrame. Because it is so light-weight, Modin provides 
speed-ups of up to 4x on a laptop with 4 physical cores.

How do I use jupyter/collab notebooks with Modin? 
""""""""""""""""""""""""""""""""""""""""""""
Just like you would use any other notebook, just replace your pandas import
with modin:

.. code-block:: python
   # import pandas as pd
   import modin.pandas as pd

Which execution engine (Ray or Dask) should I use for Modin?
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Whichever one you want! Modin uses Ray_ or Dask_ to provide an effortless way to speed up 
your pandas notebooks, scripts, and libraries. Modin will automatically detect which engine you have 
installed and use that for scheduling computation. If you want to choose a specific 
compute engine to run on, you can set the environment variable ``MODIN_ENGINE`` and 
Modin will do computation with that engine:

.. code-block:: bash
   pip install "modin[ray]" # Install Modin dependencies and Ray to run on Ray
   export MODIN_ENGINE=ray  # Modin will use Ray

   pip install "modin[dask]" # Install Modin dependencies and Dask to run on Dask
   export MODIN_ENGINE=dask  # Modin will use Dask

We also have an experimental OmniSciDB-based engine of Modin you can read about :doc:`here </developer/using_omnisci>`.
We plan to support more execution backends in future. If you have a specific request, 
please post on the #feature-requests channel on our Slack_ community. 

How can I contribute to Modin?
"""""""""""""""""""""""""""""""
**Modin is currently under active development. Requests and contributions are welcome!**

If you are interested in contributions please check out the :doc:`Getting Started</getting_started/index>`
guide then refer to the :doc:`Developer Documentation</developer/index>` section,
where you can find system architecture, internal implementation details, and other useful information.
Also check out the `Github`_ to view open issues and make contributions.

.. _issue: https://github.com/modin-project/modin/issues
.. _Dataframe: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html
.. _Slack: https://modin.org/slack.html
.. _Github: https://github.com/modin-project/modin
.. _Ray: https://github.com/ray-project/ray/
.. _Dask: https://dask.org/
