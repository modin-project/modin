Modin
=====

.. image:: https://travis-ci.com/modin-project/modin.svg?branch=master
    :target: https://travis-ci.com/modin-project/modin

.. image:: https://readthedocs.org/projects/modin/badge/?version=latest
    :target: https://modin.readthedocs.io/en/latest/?badge=latest

|

*Modin is the parent project of Pandas on Ray*

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
| import pandas as pd                          | import modin.dataframe as pd                    |
|                                              |                                                 |
| df = pd.DataFrame({'col1': [1, 2, 3],        | df = pd.DataFrame({'col1': [1, 2, 3],           |
|                    'col2': [1.0, 2.0, 3.0]}) |                    'col2': [1.0, 2.0, 3.0]})    |
|                                              |                                                 |
| df.sum()                                     | df.sum()                                        |
| ...                                          | ...                                             |
+----------------------------------------------+-------------------------------------------------+

**Pandas on Ray is currently for experimental use only. Requests and contributions are welcome!**
