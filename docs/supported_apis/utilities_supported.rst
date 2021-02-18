pandas Utilities Supported
==========================

If you ``import modin.pandas as pd`` the following operations are available from
``pd.<op>``, e.g. ``pd.concat``. If you do not see an operation that pandas enables and
would like to request it, feel free to `open an issue`_. Make sure you tell us your
primary use-case so we can make it happen faster!

The following table is structured as follows: The first column contains the method name.
The second column is a flag for whether or not there is an implementation in Modin for
the method in the left column. ``Y`` stands for yes, ``N`` stands for no, ``P`` stands
for partial (meaning some parameters may not be supported yet), and ``D`` stands for
default to pandas.

+---------------------------+---------------------------------+----------------------------------------------------+
| Utility method            | Modin Implementation? (Y/N/P/D) | Notes for Current implementation                   |
+---------------------------+---------------------------------+----------------------------------------------------+
| `pd.concat`_              | Y                               |                                                    |
+---------------------------+---------------------------------+----------------------------------------------------+
| `pd.eval`_                | Y                               |                                                    |
+---------------------------+---------------------------------+----------------------------------------------------+
| `pd.unique`_              | Y                               |                                                    |
+---------------------------+---------------------------------+----------------------------------------------------+
| ``pd.value_counts``       | Y                               | The indices order of resulting object may differ   |
|                           |                                 | from pandas.                                       |
+---------------------------+---------------------------------+----------------------------------------------------+
| `pd.cut`_                 | D                               |                                                    |
+---------------------------+---------------------------------+----------------------------------------------------+
| `pd.to_numeric`_          | D                               |                                                    |
+---------------------------+---------------------------------+----------------------------------------------------+
| `pd.factorize`_           | D                               |                                                    |
+---------------------------+---------------------------------+----------------------------------------------------+
| `pd.qcut`_                | D                               |                                                    |
+---------------------------+---------------------------------+----------------------------------------------------+
| ``pd.match``              | D                               |                                                    |
+---------------------------+---------------------------------+----------------------------------------------------+
| `pd.to_datetime`_         | D                               |                                                    |
+---------------------------+---------------------------------+----------------------------------------------------+
| `pd.get_dummies`_         | Y                               |                                                    |
+---------------------------+---------------------------------+----------------------------------------------------+
| `pd.date_range`_          | D                               |                                                    |
+---------------------------+---------------------------------+----------------------------------------------------+
| `pd.bdate_range`_         | D                               |                                                    |
+---------------------------+---------------------------------+----------------------------------------------------+
| `pd.to_timedelta`_        | D                               |                                                    |
+---------------------------+---------------------------------+----------------------------------------------------+
| ``pd.options``            | Y                               |                                                    |
+---------------------------+---------------------------------+----------------------------------------------------+
| ``pd.datetime``           | D                               |                                                    |
+---------------------------+---------------------------------+----------------------------------------------------+

Other objects & structures
--------------------------

This list is a list of objects not currently distributed by Modin. All of these objects
are compatible with the distributed components of Modin. If you are interested in
contributing a distributed version of any of these objects, feel free to open a
`pull request`_.

* Panel
* Index
* MultiIndex
* CategoricalIndex
* DatetimeIndex
* Timedelta
* Timestamp
* NaT
* PeriodIndex
* Categorical
* Interval
* UInt8Dtype
* UInt16Dtype
* UInt32Dtype
* UInt64Dtype
* SparseDtype
* Int8Dtype
* Int16Dtype
* Int32Dtype
* Int64Dtype
* CategoricalDtype
* DatetimeTZDtype
* IntervalDtype
* PeriodDtype
* RangeIndex
* Int64Index
* UInt64Index
* Float64Index
* TimedeltaIndex
* IntervalIndex
* IndexSlice
* TimeGrouper
* Grouper
* array
* Period
* DateOffset
* ExcelWriter
* SparseArray
* SparseSeries
* SparseDataFrame

.. _open an issue: https://github.com/modin-project/modin/issues
.. _pull request: https://github.com/modin-project/modin/pulls
.. _`pd.concat`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.concat.html#pandas.concat
.. _`pd.eval`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.eval.html#pandas.eval
.. _`pd.unique`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.unique.html#pandas.unique
.. _`pd.cut`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.cut.html#pandas.cut
.. _`pd.to_numeric`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.to_numeric.html#pandas.to_numeric
.. _`pd.factorize`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.factorize.html#pandas.factorize
.. _`pd.qcut`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.qcut.html#pandas.qcut
.. _`pd.to_datetime`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.to_datetime.html#pandas.to_datetime
.. _`pd.get_dummies`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.get_dummies.html#pandas.get_dummies
.. _`pd.date_range`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.date_range.html#pandas.date_range
.. _`pd.bdate_range`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.bdate_range.html#pandas.bdate_range
.. _`pd.to_timedelta`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.to_timedelta.html#pandas.to_timedelta
