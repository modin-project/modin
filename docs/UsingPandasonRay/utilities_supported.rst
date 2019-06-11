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
| ``pd.concat``             | Y                               |                                                    |
+---------------------------+---------------------------------+----------------------------------------------------+
| ``pd.eval``               | Y                               |                                                    |
+---------------------------+---------------------------------+----------------------------------------------------+
| ``pd.unique``             | D                               |                                                    |
+---------------------------+---------------------------------+----------------------------------------------------+
| ``pd.value_counts``       | D                               |                                                    |
+---------------------------+---------------------------------+----------------------------------------------------+
| ``pd.cut``                | D                               |                                                    |
+---------------------------+---------------------------------+----------------------------------------------------+
| ``pd.to_numeric``         | D                               |                                                    |
+---------------------------+---------------------------------+----------------------------------------------------+
| ``pd.factorize``          | D                               |                                                    |
+---------------------------+---------------------------------+----------------------------------------------------+
| ``pd.qcut``               | D                               |                                                    |
+---------------------------+---------------------------------+----------------------------------------------------+
| ``pd.match``              | D                               |                                                    |
+---------------------------+---------------------------------+----------------------------------------------------+
| ``pd.to_datetime``        | D                               |                                                    |
+---------------------------+---------------------------------+----------------------------------------------------+
| ``pd.get_dummies``        | Y                               |                                                    |
+---------------------------+---------------------------------+----------------------------------------------------+
| ``pd.date_range``         | D                               |                                                    |
+---------------------------+---------------------------------+----------------------------------------------------+
| ``pd.bdate_range``        | D                               |                                                    |
+---------------------------+---------------------------------+----------------------------------------------------+
| ``pd.to_timedelta``       | D                               |                                                    |
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
