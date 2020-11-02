Modin SQL API
=============

**Modin's SQL API is currently a conceptual plan, Coming Soon!**

Plans for future development
----------------------------

Our plans with the SQL API for Modin are to create an interface that allows you to
intermix SQL and pandas operations without copying the entire dataset into a new
structure between the two. This is possible due to the architecture of Modin. Currently,
Modin has a query compiler that acts as an intermediate layer between the query language
(e.g. SQL, pandas) and the execution (See architecture_ documentation for details).

*We have implemented a simple example that can be found below. Feedback welcome!*

.. code-block:: python

   >>> import modin.sql as sql
   >>>
   >>> conn = sql.connect("db_name")
   >>> c = conn.cursor()
   >>> c.execute("CREATE TABLE example (col1, col2, column 3, col4)")
   >>> c.execute("INSERT INTO example VALUES ('1', 2.0, 'A String of information', True)")
     col1  col2                 column 3  col4
   0    1   2.0  A String of information  True

   >>> c.execute("INSERT INTO example VALUES ('6', 17.0, 'A String of different information', False)")
     col1  col2                           column 3   col4
   0    1   2.0            A String of information   True
   1    6  17.0  A String of different information  False

.. _architecture: https://modin.readthedocs.io/en/latest/developer/architecture.html
