SQL on Ray
==========

**SQL on Ray is currently under development. Coming Soon!**

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
