# Licensed to Modin Development Team under one or more contributor license agreements.
# See the NOTICE file distributed with this work for additional information regarding
# copyright ownership.  The Modin Development Team licenses this file to you under the
# Apache License, Version 2.0 (the "License"); you may not use this file except in
# compliance with the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under
# the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.

"""
Implement Series's accessors public API as Pandas does.

Accessors: `Series.cat`, `Series.str`, `Series.dt`

Almost all docstrings for public and magic methods should be inherited from Pandas
for better maintability. So some codes are ignored in pydocstyle check:
    - D101: missing docstring in class
    - D102: missing docstring in public method
Manually add documentation for methods which are not presented in pandas.
"""

import sys
import numpy as np
import pandas
from modin.utils import _inherit_docstrings
from .series import Series

if sys.version_info[0] == 3 and sys.version_info[1] >= 7:
    # Python >= 3.7
    from re import Pattern as _pattern_type
else:
    # Python <= 3.6
    from re import _pattern_type


@_inherit_docstrings(
    pandas.core.arrays.categorical.CategoricalAccessor,
    excluded=[
        pandas.core.arrays.categorical.CategoricalAccessor.__init__,
    ],
)
class CategoryMethods(object):
    def __init__(self, series):
        self._series = series
        self._query_compiler = series._query_compiler

    @property
    def categories(self):
        return self._series._default_to_pandas(pandas.Series.cat).categories

    @categories.setter
    def categories(self, categories):
        def set_categories(series, categories):
            series.cat.categories = categories

        self._series._default_to_pandas(set_categories, categories=categories)

    @property
    def ordered(self):
        return self._series._default_to_pandas(pandas.Series.cat).ordered

    @property
    def codes(self):
        return Series(query_compiler=self._query_compiler.cat_codes())

    def rename_categories(self, new_categories, inplace=False):
        return self._default_to_pandas(
            pandas.Series.cat.rename_categories, new_categories, inplace=inplace
        )

    def reorder_categories(self, new_categories, ordered=None, inplace=False):
        return self._default_to_pandas(
            pandas.Series.cat.reorder_categories,
            new_categories,
            ordered=ordered,
            inplace=inplace,
        )

    def add_categories(self, new_categories, inplace=False):
        return self._default_to_pandas(
            pandas.Series.cat.add_categories, new_categories, inplace=inplace
        )

    def remove_categories(self, removals, inplace=False):
        return self._default_to_pandas(
            pandas.Series.cat.remove_categories, removals, inplace=inplace
        )

    def remove_unused_categories(self, inplace=False):
        return self._default_to_pandas(
            pandas.Series.cat.remove_unused_categories, inplace=inplace
        )

    def set_categories(self, new_categories, ordered=None, rename=False, inplace=False):
        return self._default_to_pandas(
            pandas.Series.cat.set_categories,
            new_categories,
            ordered=ordered,
            rename=rename,
            inplace=inplace,
        )

    def as_ordered(self, inplace=False):
        return self._default_to_pandas(pandas.Series.cat.as_ordered, inplace=inplace)

    def as_unordered(self, inplace=False):
        return self._default_to_pandas(pandas.Series.cat.as_unordered, inplace=inplace)

    def _default_to_pandas(self, op, *args, **kwargs):
        return self._series._default_to_pandas(
            lambda series: op(series.cat, *args, **kwargs)
        )


@_inherit_docstrings(
    pandas.core.strings.StringMethods,
    excluded=[
        pandas.core.strings.StringMethods.__init__,
    ],
)
class StringMethods(object):
    def __init__(self, series):
        # Check if dtypes is objects

        self._series = series
        self._query_compiler = series._query_compiler

    def casefold(self):
        return self._default_to_pandas(pandas.Series.str.casefold)

    def cat(self, others=None, sep=None, na_rep=None, join=None):
        if isinstance(others, Series):
            others = others._to_pandas()
        return self._default_to_pandas(
            pandas.Series.str.cat, others=others, sep=sep, na_rep=na_rep, join=join
        )

    def decode(self, encoding, errors="strict"):
        return self._default_to_pandas(
            pandas.Series.str.decode, encoding, errors=errors
        )

    def split(self, pat=None, n=-1, expand=False):
        if not pat and pat is not None:
            raise ValueError("split() requires a non-empty pattern match.")

        if expand:
            return self._default_to_pandas(
                pandas.Series.str.split, pat=pat, n=n, expand=expand
            )
        else:
            return Series(
                query_compiler=self._query_compiler.str_split(
                    pat=pat, n=n, expand=expand
                )
            )

    def rsplit(self, pat=None, n=-1, expand=False):
        if not pat and pat is not None:
            raise ValueError("rsplit() requires a non-empty pattern match.")

        if expand:
            return self._default_to_pandas(
                pandas.Series.str.rsplit, pat=pat, n=n, expand=expand
            )
        else:
            return Series(
                query_compiler=self._query_compiler.str_rsplit(
                    pat=pat, n=n, expand=expand
                )
            )

    def get(self, i):
        return Series(query_compiler=self._query_compiler.str_get(i))

    def join(self, sep):
        if sep is None:
            raise AttributeError("'NoneType' object has no attribute 'join'")
        return Series(query_compiler=self._query_compiler.str_join(sep))

    def get_dummies(self, sep="|"):
        return self._default_to_pandas(pandas.Series.str.get_dummies, sep=sep)

    def contains(self, pat, case=True, flags=0, na=np.NaN, regex=True):
        if pat is None and not case:
            raise AttributeError("'NoneType' object has no attribute 'upper'")
        return Series(
            query_compiler=self._query_compiler.str_contains(
                pat, case=case, flags=flags, na=na, regex=regex
            )
        )

    def replace(self, pat, repl, n=-1, case=None, flags=0, regex=True):
        if not (isinstance(repl, str) or callable(repl)):
            raise TypeError("repl must be a string or callable")
        return Series(
            query_compiler=self._query_compiler.str_replace(
                pat, repl, n=n, case=case, flags=flags, regex=regex
            )
        )

    def repeats(self, repeats):
        return Series(query_compiler=self._query_compiler.str_repeats(repeats))

    def pad(self, width, side="left", fillchar=" "):
        if len(fillchar) != 1:
            raise TypeError("fillchar must be a character, not str")
        return Series(
            query_compiler=self._query_compiler.str_pad(
                width, side=side, fillchar=fillchar
            )
        )

    def center(self, width, fillchar=" "):
        if len(fillchar) != 1:
            raise TypeError("fillchar must be a character, not str")
        return Series(
            query_compiler=self._query_compiler.str_center(width, fillchar=fillchar)
        )

    def ljust(self, width, fillchar=" "):
        if len(fillchar) != 1:
            raise TypeError("fillchar must be a character, not str")
        return Series(
            query_compiler=self._query_compiler.str_ljust(width, fillchar=fillchar)
        )

    def rjust(self, width, fillchar=" "):
        if len(fillchar) != 1:
            raise TypeError("fillchar must be a character, not str")
        return Series(
            query_compiler=self._query_compiler.str_rjust(width, fillchar=fillchar)
        )

    def zfill(self, width):
        return Series(query_compiler=self._query_compiler.str_zfill(width))

    def wrap(self, width, **kwargs):
        if width <= 0:
            raise ValueError("invalid width {} (must be > 0)".format(width))
        return Series(query_compiler=self._query_compiler.str_wrap(width, **kwargs))

    def slice(self, start=None, stop=None, step=None):
        if step == 0:
            raise ValueError("slice step cannot be zero")
        return Series(
            query_compiler=self._query_compiler.str_slice(
                start=start, stop=stop, step=step
            )
        )

    def slice_replace(self, start=None, stop=None, repl=None):
        return Series(
            query_compiler=self._query_compiler.str_slice_replace(
                start=start, stop=stop, repl=repl
            )
        )

    def count(self, pat, flags=0, **kwargs):
        if not isinstance(pat, (str, _pattern_type)):
            raise TypeError("first argument must be string or compiled pattern")
        return Series(
            query_compiler=self._query_compiler.str_count(pat, flags=flags, **kwargs)
        )

    def startswith(self, pat, na=np.NaN):
        return Series(query_compiler=self._query_compiler.str_startswith(pat, na=na))

    def encode(self, encoding, errors="strict"):
        return self._default_to_pandas(
            pandas.Series.str.encode, encoding, errors=errors
        )

    def endswith(self, pat, na=np.NaN):
        return Series(query_compiler=self._query_compiler.str_endswith(pat, na=na))

    def findall(self, pat, flags=0, **kwargs):
        if not isinstance(pat, (str, _pattern_type)):
            raise TypeError("first argument must be string or compiled pattern")
        return Series(
            query_compiler=self._query_compiler.str_findall(pat, flags=flags, **kwargs)
        )

    def match(self, pat, case=True, flags=0, na=np.NaN):
        if not isinstance(pat, (str, _pattern_type)):
            raise TypeError("first argument must be string or compiled pattern")
        return Series(
            query_compiler=self._query_compiler.str_match(pat, flags=flags, na=na)
        )

    def extract(self, pat, flags=0, expand=True):
        return self._default_to_pandas(
            pandas.Series.str.extract, pat, flags=flags, expand=expand
        )

    def extractall(self, pat, flags=0):
        return self._default_to_pandas(pandas.Series.str.extractall, pat, flags=flags)

    def len(self):
        return Series(query_compiler=self._query_compiler.str_len())

    def strip(self, to_strip=None):
        return Series(query_compiler=self._query_compiler.str_strip(to_strip=to_strip))

    def rstrip(self, to_strip=None):
        return Series(query_compiler=self._query_compiler.str_rstrip(to_strip=to_strip))

    def lstrip(self, to_strip=None):
        return Series(query_compiler=self._query_compiler.str_lstrip(to_strip=to_strip))

    def partition(self, sep=" ", expand=True):
        if sep is not None and len(sep) == 0:
            raise ValueError("empty separator")

        if expand:
            return self._default_to_pandas(
                pandas.Series.str.partition, sep=sep, expand=expand
            )
        else:
            return Series(
                query_compiler=self._query_compiler.str_partition(
                    sep=sep, expand=expand
                )
            )

    def repeat(self, repeats):
        return self._default_to_pandas(pandas.Series.str.repeat, repeats)

    def rpartition(self, sep=" ", expand=True):
        if sep is not None and len(sep) == 0:
            raise ValueError("empty separator")

        if expand:
            return self._default_to_pandas(
                pandas.Series.str.rpartition, sep=sep, expand=expand
            )
        else:
            return Series(
                query_compiler=self._query_compiler.str_rpartition(
                    sep=sep, expand=expand
                )
            )

    def lower(self):
        return Series(query_compiler=self._query_compiler.str_lower())

    def upper(self):
        return Series(query_compiler=self._query_compiler.str_upper())

    def title(self):
        return Series(query_compiler=self._query_compiler.str_title())

    def find(self, sub, start=0, end=None):
        if not isinstance(sub, str):
            raise TypeError(
                "expected a string object, not {0}".format(type(sub).__name__)
            )
        return Series(
            query_compiler=self._query_compiler.str_find(sub, start=start, end=end)
        )

    def rfind(self, sub, start=0, end=None):
        if not isinstance(sub, str):
            raise TypeError(
                "expected a string object, not {0}".format(type(sub).__name__)
            )
        return Series(
            query_compiler=self._query_compiler.str_rfind(sub, start=start, end=end)
        )

    def index(self, sub, start=0, end=None):
        if not isinstance(sub, str):
            raise TypeError(
                "expected a string object, not {0}".format(type(sub).__name__)
            )
        return Series(
            query_compiler=self._query_compiler.str_index(sub, start=start, end=end)
        )

    def rindex(self, sub, start=0, end=None):
        if not isinstance(sub, str):
            raise TypeError(
                "expected a string object, not {0}".format(type(sub).__name__)
            )
        return Series(
            query_compiler=self._query_compiler.str_rindex(sub, start=start, end=end)
        )

    def capitalize(self):
        return Series(query_compiler=self._query_compiler.str_capitalize())

    def swapcase(self):
        return Series(query_compiler=self._query_compiler.str_swapcase())

    def normalize(self, form):
        return Series(query_compiler=self._query_compiler.str_normalize(form))

    def translate(self, table):
        return Series(query_compiler=self._query_compiler.str_translate(table))

    def isalnum(self):
        return Series(query_compiler=self._query_compiler.str_isalnum())

    def isalpha(self):
        return Series(query_compiler=self._query_compiler.str_isalpha())

    def isdigit(self):
        return Series(query_compiler=self._query_compiler.str_isdigit())

    def isspace(self):
        return Series(query_compiler=self._query_compiler.str_isspace())

    def islower(self):
        return Series(query_compiler=self._query_compiler.str_islower())

    def isupper(self):
        return Series(query_compiler=self._query_compiler.str_isupper())

    def istitle(self):
        return Series(query_compiler=self._query_compiler.str_istitle())

    def isnumeric(self):
        return Series(query_compiler=self._query_compiler.str_isnumeric())

    def isdecimal(self):
        return Series(query_compiler=self._query_compiler.str_isdecimal())

    def _default_to_pandas(self, op, *args, **kwargs):
        return self._series._default_to_pandas(
            lambda series: op(series.str, *args, **kwargs)
        )


@_inherit_docstrings(
    pandas.core.indexes.accessors.CombinedDatetimelikeProperties,
    excluded=[
        pandas.core.indexes.accessors.CombinedDatetimelikeProperties.__init__,
    ],
)
class DatetimeProperties(object):
    def __init__(self, series):
        self._series = series
        self._query_compiler = series._query_compiler

    @property
    def date(self):
        return Series(query_compiler=self._query_compiler.dt_date())

    @property
    def time(self):
        return Series(query_compiler=self._query_compiler.dt_time())

    @property
    def timetz(self):
        return Series(query_compiler=self._query_compiler.dt_timetz())

    @property
    def year(self):
        return Series(query_compiler=self._query_compiler.dt_year())

    @property
    def month(self):
        return Series(query_compiler=self._query_compiler.dt_month())

    @property
    def day(self):
        return Series(query_compiler=self._query_compiler.dt_day())

    @property
    def hour(self):
        return Series(query_compiler=self._query_compiler.dt_hour())

    @property
    def minute(self):
        return Series(query_compiler=self._query_compiler.dt_minute())

    @property
    def second(self):
        return Series(query_compiler=self._query_compiler.dt_second())

    @property
    def microsecond(self):
        return Series(query_compiler=self._query_compiler.dt_microsecond())

    @property
    def nanosecond(self):
        return Series(query_compiler=self._query_compiler.dt_nanosecond())

    @property
    def week(self):
        return Series(query_compiler=self._query_compiler.dt_week())

    @property
    def weekofyear(self):
        return Series(query_compiler=self._query_compiler.dt_weekofyear())

    @property
    def dayofweek(self):
        return Series(query_compiler=self._query_compiler.dt_dayofweek())

    @property
    def weekday(self):
        return Series(query_compiler=self._query_compiler.dt_weekday())

    @property
    def dayofyear(self):
        return Series(query_compiler=self._query_compiler.dt_dayofyear())

    @property
    def quarter(self):
        return Series(query_compiler=self._query_compiler.dt_quarter())

    @property
    def is_month_start(self):
        return Series(query_compiler=self._query_compiler.dt_is_month_start())

    @property
    def is_month_end(self):
        return Series(query_compiler=self._query_compiler.dt_is_month_end())

    @property
    def is_quarter_start(self):
        return Series(query_compiler=self._query_compiler.dt_is_quarter_start())

    @property
    def is_quarter_end(self):
        return Series(query_compiler=self._query_compiler.dt_is_quarter_end())

    @property
    def is_year_start(self):
        return Series(query_compiler=self._query_compiler.dt_is_year_start())

    @property
    def is_year_end(self):
        return Series(query_compiler=self._query_compiler.dt_is_year_end())

    @property
    def is_leap_year(self):
        return Series(query_compiler=self._query_compiler.dt_is_leap_year())

    @property
    def daysinmonth(self):
        return Series(query_compiler=self._query_compiler.dt_daysinmonth())

    @property
    def days_in_month(self):
        return Series(query_compiler=self._query_compiler.dt_days_in_month())

    @property
    def tz(self):
        return self._query_compiler.dt_tz().to_pandas().squeeze()

    @property
    def freq(self):
        return self._query_compiler.dt_freq().to_pandas().squeeze()

    def to_period(self, *args, **kwargs):
        return Series(query_compiler=self._query_compiler.dt_to_period(*args, **kwargs))

    def to_pydatetime(self):
        return Series(query_compiler=self._query_compiler.dt_to_pydatetime()).to_numpy()

    def tz_localize(self, *args, **kwargs):
        return Series(
            query_compiler=self._query_compiler.dt_tz_localize(*args, **kwargs)
        )

    def tz_convert(self, *args, **kwargs):
        return Series(
            query_compiler=self._query_compiler.dt_tz_convert(*args, **kwargs)
        )

    def normalize(self, *args, **kwargs):
        return Series(query_compiler=self._query_compiler.dt_normalize(*args, **kwargs))

    def strftime(self, *args, **kwargs):
        return Series(query_compiler=self._query_compiler.dt_strftime(*args, **kwargs))

    def round(self, *args, **kwargs):
        return Series(query_compiler=self._query_compiler.dt_round(*args, **kwargs))

    def floor(self, *args, **kwargs):
        return Series(query_compiler=self._query_compiler.dt_floor(*args, **kwargs))

    def ceil(self, *args, **kwargs):
        return Series(query_compiler=self._query_compiler.dt_ceil(*args, **kwargs))

    def month_name(self, *args, **kwargs):
        return Series(
            query_compiler=self._query_compiler.dt_month_name(*args, **kwargs)
        )

    def day_name(self, *args, **kwargs):
        return Series(query_compiler=self._query_compiler.dt_day_name(*args, **kwargs))

    def total_seconds(self, *args, **kwargs):
        return Series(
            query_compiler=self._query_compiler.dt_total_seconds(*args, **kwargs)
        )

    def to_pytimedelta(self):
        return self._query_compiler.default_to_pandas(
            lambda df: pandas.Series.dt.to_pytimedelta(df.squeeze(axis=1).dt)
        )

    @property
    def seconds(self):
        return Series(query_compiler=self._query_compiler.dt_seconds())

    @property
    def days(self):
        return Series(query_compiler=self._query_compiler.dt_days())

    @property
    def microseconds(self):
        return Series(query_compiler=self._query_compiler.dt_microseconds())

    @property
    def nanoseconds(self):
        return Series(query_compiler=self._query_compiler.dt_nanoseconds())

    @property
    def components(self):
        from .dataframe import DataFrame

        return DataFrame(query_compiler=self._query_compiler.dt_components())

    @property
    def qyear(self):
        return Series(query_compiler=self._query_compiler.dt_qyear())

    @property
    def start_time(self):
        return Series(query_compiler=self._query_compiler.dt_start_time())

    @property
    def end_time(self):
        return Series(query_compiler=self._query_compiler.dt_end_time())

    def to_timestamp(self, *args, **kwargs):
        return Series(
            query_compiler=self._query_compiler.dt_to_timestamp(*args, **kwargs)
        )
