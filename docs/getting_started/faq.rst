Frequently Asked Questions (FAQs)
=================================

Below, you will find answers to the most commonly asked questions about
Modin. If you still cannot find the answer you are looking for, please post your
question on the #support channel on our Slack_ community or open a Github issue_.

FAQs: Why choose Modin?
-----------------------

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
often has less memory overhead than pandas. See :doc:` Why Modin? </getting_started/why_modin/pandas>`
page to learn more about how Modin is different from pandas.

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
see :doc:`out-of-memory support </getting_started/why_modin/out_of_core>` for Modin.

How does Modin compare to Dask DataFrame and Koalas?
""""""""""""""""""""""""""""""""""""""""""""""""""""

TLDR: Modin has better coverage of the pandas API, has a flexible backend, better ordering semantics,
and supports both row and column-parallel operations.
Check out :doc:`Modin vs Dask vs Koalas </getting_started/why_modin/modin_vs_dask_vs_koalas>` page detailing
the differences!

How does Modin work under the hood?
"""""""""""""""""""""""""""""""""""

Modin is logically separated into different layers that represent the hierarchy of a
typical Database Management System. User queries which perform data transformation,
data ingress or data egress pass through the Modin Query Compiler which translates
queries from the top-level pandas API Layer that users interact with to the Modin Core
Dataframe layer.
The Modin Core DataFrame is our efficient DataFrame implementation that utilizes a partitioning schema
which allows for distributing tasks and queries. From here, the Modin DataFrame works with engines like
Ray, Dask or Unidist to execute computation, and then return the results to the user.

For more details, take a look at our system :doc:`architecture </development/architecture>`.

FAQs: How to use Modin?
-----------------------

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

Which execution engine (Ray, Dask or Unidist) should I use for Modin?
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

Modin lets you effortlessly speed up your pandas workflows with either Ray_'s, Dask_'s or Unidist_'s execution engine.
You don't need to know anything about either engine in order to use it with Modin. If you only have one engine
installed, Modin will automatically detect which engine you have installed and use that for scheduling computation.
If you don't have a preference, we recommend starting with Modin's default Ray engine.
If you want to use a specific compute engine, you can set the environment variable ``MODIN_ENGINE``
and Modin will do computation with that engine:

.. code-block:: bash

    pip install "modin[ray]" # Install Modin dependencies and Ray to run on Ray
    export MODIN_ENGINE=ray  # Modin will use Ray

    pip install "modin[dask]" # Install Modin dependencies and Dask to run on Dask
    export MODIN_ENGINE=dask  # Modin will use Dask

    pip install "modin[mpi]" # Install Modin dependencies and MPI to run on MPI through unidist.
    export MODIN_ENGINE=unidist  # Modin will use Unidist
    export UNIDIST_BACKEND=mpi   # Unidist will use MPI backend.

This can also be done with:

.. code-block:: python

    import modin.config as modin_cfg
    import unidist.config as unidist_cfg

    modin_cfg.Engine.put("ray")  # Modin will use Ray
    modin_cfg.Engine.put("dask")  # Modin will use Dask

    modin_cfg.Engine.put('unidist') # Modin will use Unidist
    unidist_cfg.Backend.put('mpi') # Unidist will use MPI backend

We plan to support more execution engines in future. If you have a specific request,
please post on the #feature-requests channel on our Slack_ community.

How do I connect Modin to a database via `read_sql`?
""""""""""""""""""""""""""""""""""""""""""""""""""""

To read from a SQL database, you have two options:

1) Pass a connection string, e.g. ``postgresql://reader:NWDMCE5xdipIjRrp@hh-pgsql-public.ebi.ac.uk:5432/pfmegrnargs``
2) Pass an open database connection, e.g. for psycopg2, ``psycopg2.connect("dbname=pfmegrnargs user=reader password=NWDMCE5xdipIjRrp host=hh-pgsql-public.ebi.ac.uk")``

The first option works with both Modin and pandas. If you try the second option
in Modin, Modin will default to pandas because open database connections cannot be pickled.
Pickling is required to send connection details to remote workers.
To handle the unique requirements of distributed database access, Modin has a distributed
database connection called ``ModinDatabaseConnection``:

.. code-block:: python

    import modin.pandas as pd
    from modin.db_conn import ModinDatabaseConnection
    con = ModinDatabaseConnection(
        'psycopg2',
        host='hh-pgsql-public.ebi.ac.uk',
        dbname='pfmegrnargs',
        user='reader',
        password='NWDMCE5xdipIjRrp')
    df = pd.read_sql("SELECT * FROM rnc_database",
            con,
            index_col=None,
            coerce_float=True,
            params=None,
            parse_dates=None,
            chunksize=None)


The ``ModinDatabaseConnection`` will save any arguments you supply it and forward
them to the workers to make their own connections.

How can I contribute to Modin?
""""""""""""""""""""""""""""""

**Modin is currently under active development. Requests and contributions are welcome!**

If you are interested in contributing please check out the :doc:`Contributing Guide</development/contributing>`
and then refer to the :doc:`Development Documentation</development/index>`,
where you can find system architecture, internal implementation details, and other useful information.
Also check out the `Github`_ to view open issues and make contributions.

.. _issue: https://github.com/modin-project/modin/issues
.. _Slack: https://join.slack.com/t/modin-project/shared_invite/zt-yvk5hr3b-f08p_ulbuRWsAfg9rMY3uA
.. _Github: https://github.com/modin-project/modin
.. _Ray: https://github.com/ray-project/ray/
.. _Dask: https://github.com/dask/dask
.. _Unidist: https://github.com/modin-project/unidist
.. _papers: https://people.eecs.berkeley.edu/~totemtang/paper/Modin.pdf
.. _guide: https://modin.readthedocs.io/en/latest/getting_started/installation.html#installing-on-google-colab
.. _tutorial: https://github.com/modin-project/modin/tree/main/examples/tutorial
