pandas Utilities Supported
==========================

If you run ``import modin.pandas as pd``, the following operations are available from
``pd.<op>``, e.g. ``pd.concat``. If you do not see an operation that pandas enables and
would like to request it, feel free to `open an issue`_. Make sure you tell us your
primary use-case so we can make it happen faster!

Utilities Supported APIs table is structured as follows: The first column contains the method name,
the second column - the parameter name of this method, and other columns contain
different flags describing particular properties of method parameters for a concrete
execution.

The flags stand for the following:

.. table::
   :widths: 1, 5

   +-------------+-----------------------------------------------------------------------------------------------+
   | Flag        | Meaning                                                                                       |
   +=============+===============================================================================================+
   | Supported   | Parameter is supported, it's usage brings performance improvement                             |
   +-------------+-----------------------------------------------------------------------------------------------+
   | Harmful     | Usage of this parameter can be harmful for performance of your application. Usually this      |
   |             | happens when parameter (full range of values and all types) is not supported and default      |
   |             | pandas implementation is used                                                                 |
   +-------------+-----------------------------------------------------------------------------------------------+
   | Non-lazy    | Usage of this parameter can trigger non-lazy execution (actual for OmniSci execution only)    |
   +-------------+-----------------------------------------------------------------------------------------------+
   | Partial     | Parameter can be partly unsupported, it's usage can be harmful for performance of your        |
   |             | appcication. This can happen if some parameter values or types are not supported (for example |
   |             | boolean values are suported while integer are not) and default pandas implementation is used  |
   +-------------+-----------------------------------------------------------------------------------------------+
   | pure pandas | Usage of this parameter, triggers usage of original pandas function as is, no performance     |
   |             | degradation/improvement should be observed                                                    |
   +-------------+-----------------------------------------------------------------------------------------------+

Parameters Notes
----------------

.. csv-table::
   :file: utilities_supported.csv
   :header-rows: 1

Other objects & structures
--------------------------

This list is a list of objects not currently distributed by Modin. All of these objects
are compatible with the distributed components of Modin. If you are interested in
contributing a distributed version of any of these objects, feel free to open a
`pull request`_.

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
