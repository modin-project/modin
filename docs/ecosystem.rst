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

.. code-block:: python

    from modin.pandas.io import to_pandas

    pandas_df = to_pandas(modin_df)

to_numpy
--------

.. code-block:: python

    from modin.pandas.io import to_numpy

    numpy_arr = to_numpy(modin_df)
