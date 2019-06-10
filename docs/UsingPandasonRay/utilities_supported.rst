pandas Utilities Supported
=========================

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
| concat                    | Y                               |                                                    |
+---------------------------+---------------------------------+----------------------------------------------------+
| eval                      | Y                               |                                                    |
+---------------------------+---------------------------------+----------------------------------------------------+
| unique                    | D                               |                                                    |
+---------------------------+---------------------------------+----------------------------------------------------+
| value_counts              | D                               |                                                    |
+---------------------------+---------------------------------+----------------------------------------------------+
| cut                       | D                               |                                                    |
+---------------------------+---------------------------------+----------------------------------------------------+
| to_numeric                | D                               |                                                    |
+---------------------------+---------------------------------+----------------------------------------------------+
|

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
