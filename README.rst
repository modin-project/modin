..class::center
Modin
=====



.. image:: https://travis-ci.com/modin-project/modin.svg?branch=master
    :target: https://travis-ci.com/modin-project/modin

.. image:: https://readthedocs.org/projects/modin/badge/?version=latest
    :target: https://modin.readthedocs.io/en/latest/?badge=latest

.. image:: https://badge.fury.io/py/modin.svg
    :target: https://badge.fury.io/py/modin

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/ambv/black
|

*Modin is a library that allows you to scale your Pandas workflows by changing one line of code*

Modin can be installed with pip: ``pip install modin``

Pandas
------

Pandas is a library that allows you to effortlessly scale pandas by changing only a single line of code



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

**Pandas on Ray is currently under active development. Requests and contributions are welcome!**



More information and Getting Involved
-------------------------------------

- `Documentation`_
- Ask questions on our mailing list `modin-dev@googlegroups.com`_.
- Submit bug reports to our `GitHub Issues Page`_.
- Contributions are welcome! Open a `pull request`_.

.. _`Documentation`: https://modin.readthedocs.io/en/latest/
.. _`modin-dev@googlegroups.com`: https://groups.google.com/forum/#!forum/modin-dev
.. _`GitHub Issues Page`: https://github.com/modin-project/modin/issues
.. _`pull request`: https://github.com/modin-project/modin/pulls
