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

*Pandas on Ray is a project designed to effortlessly scale your pandas code by changing only a single line of code*

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

More information and Getting Involved
-------------------------------------

- `Documentation`_
- Ask questions on our mailing list `modin-dev@googlegroups.com`_.
- Please report bugs by submitting a `GitHub issue`_.
- Submit contributions using `pull requests`_.

.. _`Documentation`: http://http://modin.readthedocs.io/en/latest/
.. _`modin-dev@googlegroups.com`: https://groups.google.com/forum/#!forum/modin-dev
.. _`GitHub issue`: https://github.com/modin-project/modin/issues
.. _`pull requests`: https://github.com/modin-project/modin/pulls
