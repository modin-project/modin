Modin Supported Methods
=======================

For your convenience, we have compiled a list of currently implemented APIs and methods
available in Modin. This documentation is updated as new methods and APIs are merged
into the master branch, and not necessarily correct as of the most recent release. In
order to install the latest version of Modin, follow the directions found on the
`installation page`_.

Questions on implementation details
-----------------------------------

If you have a question about the implementation details or would like more information
about an API or method in Modin, please contact the Modin `developer mailing list`_.

API Completeness
----------------

.. raw:: html

   <img src="https://img.shields.io/badge/pandas%20api%20coverage-71.77%25-orange.svg">

Currently, we support ~71% of the pandas API. The exact methods we have implemented are
listed below.

We have taken a community-driven approach to implementing new methods. We did a `study
on pandas usage`_ to learn what the most-used APIs are. Modin currently supports **93%**
of the pandas API based on our study of pandas usage, and we are actively expanding the
API.

Defaulting to pandas
--------------------

The remaining unimplemented methods default to pandas. This allows users to continue
using Modin even though their workloads contain functions not yet implemented in Modin.
Here is a diagram of how we convert to pandas and perform the operation:

.. image:: img/convert_to_pandas.png
   :align: center

We first convert to a pandas DataFrame, then perform the operation. There is a
performance penalty for going from a partitioned Modin DataFrame to pandas because of
the communication cost and single-threaded nature of pandas. Once the pandas operation
has completed, we convert the DataFrame back into a partitioned Modin DataFrame. This
way, operations performed after something defaults to pandas will be optimized with
Modin.

DataFrame
---------

Please see the `DataFrame supported`_ page to view what APIs are currently implemented.
If you have need of an operation that is listed as not implemented, feel free to open an
issue on the `GitHub repository`_. Contributions are also welcome!


Series
------

Currently, whenever a Series is used or returned, we use a pandas Series. In the future,
we're going to implement a distributed Series, but until then there will be some
performance bottlenecks. The pandas Series is completely compatible with all operations
that both require and return one in Modin.

IO
--

A number of IO methods default to pandas. We have parallelized ``read_csv`` and
``read_parquet``, though many of the remaining methods can be relatively easily
parallelized. Some of the operations default to the pandas implementation, meaning it
will read in serially as a single, non-distributed DataFrame and distribute it.
Performance will be affected by this.

+--------------------+--------------------+----------------------------------------------------+
| IO method          | Implemented?       | Limitations/Notes for Current implementation       |
+--------------------+--------------------+----------------------------------------------------+
| ``read_csv``       | Y                  |                                                    |
+--------------------+--------------------+----------------------------------------------------+
| ``read_table``     | Y                  |                                                    |
+--------------------+--------------------+----------------------------------------------------+
| ``read_parquet``   | Y                  |                                                    |
+--------------------+--------------------+----------------------------------------------------+
| ``read_json``      | Y                  | Defaults to pandas implementation                  |
+--------------------+--------------------+----------------------------------------------------+
| ``read_html``      | Y                  | Defaults to pandas implementation                  |
+--------------------+--------------------+----------------------------------------------------+
| ``read_clipboard`` | Y                  | Defaults to pandas implementation                  |
+--------------------+--------------------+----------------------------------------------------+
| ``read_excel``     | Y                  | Defaults to pandas implementation                  |
+--------------------+--------------------+----------------------------------------------------+
| ``read_hdf``       | Y                  |                                                    |
+--------------------+--------------------+----------------------------------------------------+
| ``read_feather``   | Y                  | Defaults to pandas implementation                  |
+--------------------+--------------------+----------------------------------------------------+
| ``read_msgpack``   | Y                  | Defaults to pandas implementation                  |
+--------------------+--------------------+----------------------------------------------------+
| ``read_stata``     | Y                  | Defaults to pandas implementation                  |
+--------------------+--------------------+----------------------------------------------------+
| ``read_sas``       | Y                  | Defaults to pandas implementation                  |
+--------------------+--------------------+----------------------------------------------------+
| ``read_pickle``    | Y                  | Defaults to pandas implementation                  |
+--------------------+--------------------+----------------------------------------------------+
| ``read_sql``       | Y                  | Defaults to pandas implementation                  |
+--------------------+--------------------+----------------------------------------------------+

List of Other Supported Operations Available on Import
------------------------------------------------------

If you ``import modin.pandas as pd`` the following operations are available from
``pd.<op>``, e.g. ``pd.concat``. If you do not see an operation that pandas enables and
would like to request it, feel free to `open an issue`_. Make sure you tell us your
primary use-case so we can make it happen faster!

* ``pd.concat``
* ``pd.eval``
* ``pd.unique``
* ``pd.value_counts``
* ``pd.cut``
* ``pd.to_numeric``
* ``pd.factorize``
* ``pd.test``
* ``pd.qcut``
* ``pd.match``
* ``pd.to_datetime``
* ``pd.get_dummies``
* ``pd.Panel``
* ``pd.date_range``
* ``pd.Index``
* ``pd.MultiIndex``
* ``pd.Series``
* ``pd.bdate_range``
* ``pd.DatetimeIndex``
* ``pd.to_timedelta``
* ``pd.set_eng_float_format``
* ``pd.set_option``
* ``pd.CategoricalIndex``
* ``pd.Timedelta``
* ``pd.Timestamp``
* ``pd.NaT``
* ``pd.PeriodIndex``
* ``pd.Categorical``

.. _`GitHub repository`: https://github.com/modin-project/modin/issues
.. _`developer mailing list`: https://groups.google.com/forum/#!forum/modin-dev
.. _`installation page`: installation.html#building-modin-from-source
.. _study on pandas usage: https://rise.cs.berkeley.edu/blog/pandas-on-ray-early-lessons/
.. _`open an issue`: https://github.com/modin-project/modin/issues
