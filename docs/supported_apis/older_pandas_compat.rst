===================================
Pandas backwards compatibility mode
===================================

Modin verions 0.16 and 0.17, but no later minor versions, had limited support
for running with legacy pandas versions. The latest version of Modin no longer
has such support.

Motivation for compatibility mode
---------------------------------

Modin aims to keep compatibility with latest pandas release, hopefully catching up each release
within a few days.

However, due to certain restrictions like need to use Python 3.6 it forces some users to
use older pandas (1.1.x for Python 3.6, specifically), which normally would mean they're
bound to be using ancient Modin as well.

To overcome this, Modin has special "compatibility mode" where some basic functionality
works, but please note that the support is "best possible effort" (e.g. not all older bugs
are worth fixing).

Known issues with pandas 1.1.x
------------------------------

* ``pd.append()`` does not preserve the order of columns in older pandas while Modin does
* ``.astype()`` produces different error type on incompatible dtypes
* ``read_csv()`` does not support reading from ZIP file *with compression* in parallel mode
* ``read_*`` do not support ``storage_option`` named argument
* ``to_csv()`` does not support binary mode for output file
* ``read_excel()`` does not support ``.xlsx`` files
* ``read_fwf()`` has a bug with list of skiprows and non-None nrows: `pandas-dev#10261`_
* ``.agg(int-value)`` produces TypeError in older pandas but Modin raises AssertionError
* ``Series.reset_index(drop=True)`` does not ignore ``name`` in older pandas while Modin ignores it
* ``.sort_index(ascending=None)`` does not raise ValueError in older pandas while Modin raises it

Please keep in mind that there are probably more issues which are not yet uncovered!

.. _`pandas-dev#10261`: https://github.com/pandas-dev/pandas/issues/10261
