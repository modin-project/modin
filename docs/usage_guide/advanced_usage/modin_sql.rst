SQL on Modin Dataframes
=======================

Modin provides a SQL API that allows you to intermix SQL and pandas operations
without copying the entire dataset into a new structure between the two. This is possible
due to the architecture of Modin. Currently, Modin has a query compiler that acts as an
intermediate layer between the query language (e.g. SQL, pandas) and the execution
(See :doc:`architecture </development/architecture>` documentation for details).

To execute SQL queries, Modin uses either dfsql third-party library or, in case of HDK
engine (See :doc:`Using HDK </development/using_hdk>` documentation for details)
the queries are executed directly by HDK. Thus, to execute SQL queries, either dfsql
or pyhdk module must be installed.


A Short Example Using the Google Play Store
""""""""""""""""""""""""""""""""""""""""""""

.. code-block:: python

    import modin.pandas as pd
    import modin.experimental.sql as sql

    # read google play app store list from csv
    gstore_apps_df = pd.read_csv("https://tinyurl.com/googleplaystorecsv")

    #add this code to prevent Nan value error
    gstore_apps_df = gstore_apps_df.dropna()

.. figure:: /img/modin_sql_google_play_table.png
    :align: center 

Imagine that you want to quickly select from ‘gstore_apps_df’ the columns 
App, Category, and Rating, where Price is ‘0’.

.. code-block:: python

    # You can then define the query that you want to perform
    query_str = "SELECT App, Category, Rating FROM gstore_apps WHERE Price = '0'"

    # And simply apply that query to a dataframe
    result_df = sql.query(query_str, gstore_apps=gstore_apps_df)

    # Or, in this case, where the query only requires one table,
    # you can also ignore the FROM part in the query string:
    sql_str = "SELECT App, Category, Rating WHERE Price = '0' "

    # DataFrame.sql() can take query strings without FROM statement
    # NOTE: this method required the dfsql module to be installed!
    result_df = gstore_apps_df.sql(sql_str)

Writing Complex Queries
"""""""""""""""""""""""

For complex queries, it's recommended to use the HDK engine because it's much more
powerful, comparing to dfsql. Especially, if multiple data frames are involved.

Let's explore a more complicated example.

.. code-block:: python

    gstore_reviews_df = pd.read_csv("https://tinyurl.com/googleplaystoreurcsv")

    #add this code to prevent Nan value error
    gstore_apps_df = gstore_apps_df.dropna()

.. figure:: /img/modin_sql_google_play_ur_table.png
    :align: center 


Say we want to retrieve the top 10 app categories ranked by best average ‘sentiment_polarity’ where the 
average ‘sentiment_subjectivity’ is less than 0.5.

Since ‘Category’ is on the **gstore_apps_df** and sentiment_polarity is on **gstore_reviews_df**, 
we need to join the two tables, and operate averages on that join.

.. code-block:: python

    # Single query with join and group by
    sql_str = """
    SELECT
    category,
    AVG(sentiment_polarity) AS avg_sentiment_polarity,
    AVG(sentiment_subjectivity) AS avg_sentiment_subjectivity
    FROM (
    SELECT
        category,
        CAST(sentiment as float) AS sentiment,
        CAST(sentiment_polarity AS float) AS sentiment_polarity,
        CAST(sentiment_subjectivity AS float) AS sentiment_subjectivity
    FROM gstore_apps_df
        INNER JOIN gstore_reviews_df
        ON gstore_apps_df.app = gstore_reviews_df.app
    ) sub
    GROUP BY category
    HAVING avg_sentiment_subjectivity < 0.5
    ORDER BY avg_sentiment_polarity DESC
    LIMIT 10
    """

    # Run query using apps and reviews dataframes, 
    # NOTE: that you simply pass the names of the tables in the query as arguments

    result_df = sql.query( sql_str, 
                            gstore_apps_df = gstore_apps_df, 
                            gstore_reviews_df = gstore_reviews_df)


Or, you can bring the best of doing this in python and run the query in multiple parts (it’s up to you). 

.. code-block:: python

    # join the items and reviews

    result_df = sql.query("""
    SELECT
        category,
        sentiment,
        sentiment_polarity,
        sentiment_subjectivity
    FROM gstore_apps_df INNER JOIN gstore_reviews_df
    ON gstore_apps_df.app = gstore_reviews_df.app""",
                          gstore_apps_df=gstore_apps_df,
                          gstore_reviews_df=gstore_reviews_df)

    # group by category and calculate averages

    result_df = sql.query("""
    SELECT
        category,
        AVG(sentiment_polarity) AS avg_sentiment_polarity,
        AVG(sentiment_subjectivity) AS avg_sentiment_subjectivity
    FROM result_df
    GROUP BY category
    HAVING CAST(avg_sentiment_subjectivity AS float) < 0.5
    ORDER BY avg_sentiment_polarity DESC
    LIMIT 10""",
    result_df=result_df)


If you have a cluster or even a computer with more than one CPU core, 
you can write SQL and Modin will run those queries in a distributed and optimized way. 

Further Examples and Full Documentation
"""""""""""""""""""""""""""""""""""""""
In the meantime, you can check out our `Example Notebook`_ that contains more 
examples and ideas, as well as this blog_ explaining Modin SQL usage.


.. _MindsDB: https://mindsdb.com/
.. _Example Notebook: https://github.com/mindsdb/dfsql/blob/stable/testdrive.ipynb
.. _blog: https://medium.com/riselab/why-every-data-scientist-using-pandas-needs-modin-bringing-sql-to-dataframes-3b216b29a7c0
