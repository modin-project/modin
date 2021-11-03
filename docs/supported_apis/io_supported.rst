``pd.read_<file>`` and I/O APIs
=================================

A number of IO methods default to pandas. We have parallelized ``read_csv`` and
``read_parquet``, though many of the remaining methods can be relatively easily
parallelized. Some of the operations default to the pandas implementation, meaning it
will read in serially as a single, non-distributed DataFrame and distribute it.
Performance will be affected by this.

The following table is structured as follows: The first column contains the method name.
The second column is a flag for whether or not there is an implementation in Modin for
the method in the left column. ``Y`` stands for yes, ``N`` stands for no, ``P`` stands
for partial (meaning some parameters may not be supported yet), and ``D`` stands for
default to pandas.

+--------------------+---------------------------------+----------------------------------------------------+
| IO method          | Modin Implementation? (Y/N/P/D) | Notes for Current implementation                   |
+--------------------+---------------------------------+----------------------------------------------------+
| `read_csv`_        | Y                               |                                                    |
+--------------------+---------------------------------+----------------------------------------------------+
| `read_table`_      | Y                               |                                                    |
+--------------------+---------------------------------+----------------------------------------------------+
| `read_parquet`_    | Y                               |                                                    |
+--------------------+---------------------------------+----------------------------------------------------+
| `read_json`_       | P                               | Implemented for ``lines=True``                     |
+--------------------+---------------------------------+----------------------------------------------------+
| `read_html`_       | D                               |                                                    |
+--------------------+---------------------------------+----------------------------------------------------+
| `read_clipboard`_  | D                               |                                                    |
+--------------------+---------------------------------+----------------------------------------------------+
| `read_excel`_      | D                               |                                                    |
+--------------------+---------------------------------+----------------------------------------------------+
| `read_hdf`_        | D                               |                                                    |
+--------------------+---------------------------------+----------------------------------------------------+
| `read_feather`_    | Y                               |                                                    |
+--------------------+---------------------------------+----------------------------------------------------+
| `read_msgpack`_    | D                               |                                                    |
+--------------------+---------------------------------+----------------------------------------------------+
| `read_stata`_      | D                               |                                                    |
+--------------------+---------------------------------+----------------------------------------------------+
| `read_sas`_        | D                               |                                                    |
+--------------------+---------------------------------+----------------------------------------------------+
| `read_pickle`_     | D                               | Experimental implementation:                       |
|                    |                                 | read_pickle_distributed                            |
+--------------------+---------------------------------+----------------------------------------------------+
| `read_sql`_        | Y                               |                                                    |
+--------------------+---------------------------------+----------------------------------------------------+

.. _`read_csv`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html#pandas.read_csv
.. _`read_table`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_table.html#pandas.read_table
.. _`read_parquet`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_parquet.html#pandas.read_parquet
.. _`read_json`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_json.html#pandas.read_json
.. _`read_html`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_html.html#pandas.read_html
.. _`read_clipboard`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_clipboard.html#pandas.read_clipboard
.. _`read_excel`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_excel.html#pandas.read_excel
.. _`read_hdf`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_hdf.html#pandas.read_hdf
.. _`read_feather`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_feather.html#pandas.read_feather
.. _`read_msgpack`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_msgpack.html#pandas.read_msgpack
.. _`read_stata`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_stata.html#pandas.read_stata
.. _`read_sas`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_sas.html#pandas.read_sas
.. _`read_pickle`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_pickle.html#pandas.read_pickle
.. _`read_sql`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_sql.html#pandas.read_sql

Connecting to a database for `read_sql`
""""""""""""""""""""""""""""""""""""""""""""
To make Pandas read from a SQL database, you have two options:

1) Pass a connection string, e.g. `postgresql://reader:NWDMCE5xdipIjRrp@hh-pgsql-public.ebi.ac.uk:5432/pfmegrnargs`
2) Pass an open database connection, e.g. for psycopg2, `psycopg2.connect("dbname=pfmegrnargs user=reader password=NWDMCE5xdipIjRrp host=hh-pgsql-public.ebi.ac.uk")`

The first option works with both Modin and Pandas. If you try the second option
in Modin, Modin, will default to Pandas, because it can't pickle the open
connection and send it to remote workers. The solution is to use a
`ModinDatabaseConnection`, like so:

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


The `ModinDatabaseConnection` will save any arguments you supply it and forward
them to the workers to make their own connections.