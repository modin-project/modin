Ecosystem
=========

There is a constantly growing number of users and packages using pandas
to address their specific needs in data preparation, analysis and visualization.
pandas is being used ubiquitously and is a good choise to handle small-sized data.
However, pandas scales poorly and is non-interactive on moderate to large datasets.
Modin provides a drop-in replacement API for pandas and scales computation across nodes and
CPUs available. What you need to do to switch to Modin is just replace a single line of code.

.. code-block:: python

    # import pandas as pd
    import modin.pandas as pd

While most packages can consume a pandas DataFrame and operate it efficiently,
this is not the case with a Modin DataFrame due to its distributed nature.
Thus, some packages may lack support for handling Modin DataFrame(s) correctly and,
moreover, efficiently. Modin implements such methods as ``__array__``, ``__dataframe__``, etc.
to facilitate other libraries to consume a Modin DataFrame. If you feel that a certain library
can operate efficiently with a specific format of data, it is possible to convert a Modin DataFrame
to the format preferred.

to_pandas
---------

You can refer to `pandas ecosystem`_ page to get more details on
where pandas can be used and what libraries it powers.

.. code-block:: python

    from modin.pandas.io import to_pandas

    pandas_df = to_pandas(modin_df)

to_numpy
--------

You can refer to `NumPy ecosystem`_ section of NumPy documentation to get more details on
where NumPy can be used and what libraries it powers.

.. code-block:: python

    from modin.pandas.io import to_numpy

    numpy_arr = to_numpy(modin_df)

to_ray
------

You can refer to `Ray Data`_ page to get more details on
where Ray Dataset can be used and what libraries it powers.

.. code-block:: python

    from modin.pandas.io import to_ray

    ray_dataset = to_ray(modin_df)

to_dask
-------

You can refer to `Dask DataFrame`_ page to get more details on
where Dask DataFrame can be used and what libraries it powers.

.. code-block:: python

    from modin.pandas.io import to_dask

    dask_df = to_dask(modin_df)

.. _pandas ecosystem: https://pandas.pydata.org/community/ecosystem.html
.. _NumPy ecosystem: https://numpy.org
.. _Ray Data: https://docs.ray.io/en/latest/data/data.html
.. _Dask DataFrame: https://docs.dask.org/en/stable/dataframe.html

