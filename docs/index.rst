Modin
=====

.. raw:: html

  <embed>
    <a href="https://github.com/modin-project/modin"><img style="position: absolute; top: 0; right: 0; border: 0;" src="https://camo.githubusercontent.com/365986a132ccd6a44c23a9169022c0b5c890c387/68747470733a2f2f73332e616d617a6f6e6177732e636f6d2f6769746875622f726962626f6e732f666f726b6d655f72696768745f7265645f6161303030302e706e67" alt="Fork me on GitHub" data-canonical-src="https://s3.amazonaws.com/github/ribbons/forkme_right_red_aa0000.png"></a>
  </embed>

*Modin is a library for unifying the way you interact with your data*

Modin can be installed with pip: ``pip install modin``

Pandas on Ray
-------------

*Pandas on Ray is a project designed to effortlessly scale your pandas code requiring a change of only a single line of code*

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

.. code-block::

  >>> import modin.sql as sql
  Process STDOUT and STDERR is being redirected to /tmp/raylogs/.
  Waiting for redis server at 127.0.0.1:46487 to respond...
  Waiting for redis server at 127.0.0.1:23966 to respond...
  Starting local scheduler with the following resources: {'GPU': 0, 'CPU': 8}.

  ======================================================================
  View the web UI at http://localhost:8892/notebooks/ray_ui78522.ipynb?token=02776ac38ddf5756b29da5b06ad06c491dc9ddca324b1f0a
  ======================================================================

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

.. toctree::
   :maxdepth: 1
   :caption: Installation

   installation.rst

.. toctree::
   :maxdepth: 1
   :caption: Pandas on Ray

   pandas_on_ray.rst
   pandas_supported.rst

.. toctree::
   :maxdepth: 1
   :caption: SQL on Ray

   sql_on_ray.rst
