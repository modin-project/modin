Modin
=====

.. image:: https://travis-ci.com/modin-project/modin.svg?branch=master
    :target: https://travis-ci.com/modin-project/modin

.. image:: https://readthedocs.org/projects/modin/badge/?version=latest
    :target: https://modin.readthedocs.io/en/latest/?badge=latest

|

*Modin is a library for unifying the way you interact with your data*

Modin can be installed with pip: ``pip install modin``

Pandas on Ray
-------------

*Pandas on Ray is a library that allows you to effortlessly scale pandas by changing only a single line of code*

+----------------------------------------------+-------------------------------------------------+
| **pandas**                                   | **Pandas on Ray**                               |
+----------------------------------------------+-------------------------------------------------+
|.. code-block:: python                        |.. code-block:: python                           |
|                                              |                                                 |
| # Normal pandas import                       | # Pandas on Ray import                          |
| import pandas as pd                          | import modin.pandas as pd                       |
|                                              |                                                 |
| df = pd.DataFrame({'col1': [1, 2, 3],        | df = pd.DataFrame({'col1': [1, 2, 3],           |
|                    'col2': [1.0, 2.0, 3.0]}) |                    'col2': [1.0, 2.0, 3.0]})    |
|                                              |                                                 |
| df.sum()                                     | df.sum()                                        |
| ...                                          | ...                                             |
+----------------------------------------------+-------------------------------------------------+

**Pandas on Ray is currently for experimental use only. Requests and contributions are welcome!**

SQL on Ray
----------

*SQL on Ray is currently under development. Coming Soon!*

**We have implemented a simple example that can be found below. Feedback welcome!**

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

More information and Getting Involved
-------------------------------------

- `Documentation`_
- Ask questions on our mailing list `modin-dev@googlegroups.com`_.
- Submit bug reports to our `GitHub Issues Page`_.
- Contributions are welcome! Open a `pull request`_.

.. _`Documentation`: http://http://modin.readthedocs.io/en/latest/
.. _`modin-dev@googlegroups.com`: https://groups.google.com/forum/#!forum/modin-dev
.. _`GitHub Issues Page`: https://github.com/modin-project/modin/issues
.. _`pull request`: https://github.com/modin-project/modin/pulls
