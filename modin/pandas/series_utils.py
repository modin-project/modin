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
Implement Series's accessors public API as pandas does.

Accessors: `Series.cat`, `Series.str`, `Series.dt`
"""

from __future__ import annotations

import re
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np
import pandas

from modin.logging import ClassLogger
from modin.utils import _inherit_docstrings

if TYPE_CHECKING:
    from datetime import tzinfo

    from pandas._typing import npt

    from modin.core.storage_formats import BaseQueryCompiler
    from modin.pandas import Series


@_inherit_docstrings(pandas.core.arrays.arrow.ListAccessor)
class ListAccessor(ClassLogger):
    _series: Series
    _query_compiler: BaseQueryCompiler

    def __init__(self, data: Series = None):
        self._series = data
        self._query_compiler = data._query_compiler

    @cached_property
    def _Series(self) -> Series:  # noqa: GL08
        # to avoid cyclic import
        from .series import Series

        return Series

    def flatten(self):
        return self._Series(query_compiler=self._query_compiler.list_flatten())

    def len(self):
        return self._Series(query_compiler=self._query_compiler.list_len())

    def __getitem__(self, key):
        return self._Series(
            query_compiler=self._query_compiler.list__getitem__(key=key)
        )


@_inherit_docstrings(pandas.core.arrays.arrow.StructAccessor)
class StructAccessor(ClassLogger):
    _series: Series
    _query_compiler: BaseQueryCompiler

    def __init__(self, data: Series = None):
        self._series = data
        self._query_compiler = data._query_compiler

    @cached_property
    def _Series(self) -> Series:  # noqa: GL08
        # to avoid cyclic import
        from modin.pandas.series import Series

        return Series

    @property
    def dtypes(self):
        return self._Series(query_compiler=self._query_compiler.struct_dtypes())

    def field(self, name_or_index):
        return self._Series(
            query_compiler=self._query_compiler.struct_field(
                name_or_index=name_or_index
            )
        )

    def explode(self):
        from modin.pandas.dataframe import DataFrame

        return DataFrame(query_compiler=self._query_compiler.struct_explode())


@_inherit_docstrings(pandas.core.arrays.categorical.CategoricalAccessor)
class CategoryMethods(ClassLogger):
    _series: Series
    _query_compiler: BaseQueryCompiler

    def __init__(self, data: Series):
        self._series = data
        self._query_compiler = data._query_compiler

    @cached_property
    def _Series(self) -> Series:  # noqa: GL08
        # to avoid cyclic import
        from modin.pandas.series import Series

        return Series

    @property
    def categories(self):
        return self._series.dtype.categories

    @categories.setter
    def categories(self, categories):
        def set_categories(series, categories):
            series.cat.categories = categories

        self._series._default_to_pandas(set_categories, categories=categories)

    @property
    def ordered(self):
        return self._series.dtype.ordered

    @property
    def codes(self):
        return self._Series(query_compiler=self._query_compiler.cat_codes())

    def rename_categories(self, new_categories):
        return self._default_to_pandas(
            pandas.Series.cat.rename_categories, new_categories
        )

    def reorder_categories(self, new_categories, ordered=None):
        return self._default_to_pandas(
            pandas.Series.cat.reorder_categories,
            new_categories,
            ordered=ordered,
        )

    def add_categories(self, new_categories):
        return self._default_to_pandas(pandas.Series.cat.add_categories, new_categories)

    def remove_categories(self, removals):
        return self._default_to_pandas(pandas.Series.cat.remove_categories, removals)

    def remove_unused_categories(self):
        return self._default_to_pandas(pandas.Series.cat.remove_unused_categories)

    def set_categories(self, new_categories, ordered=None, rename=False):
        return self._default_to_pandas(
            pandas.Series.cat.set_categories,
            new_categories,
            ordered=ordered,
            rename=rename,
        )

    def as_ordered(self):
        return self._default_to_pandas(pandas.Series.cat.as_ordered)

    def as_unordered(self):
        return self._default_to_pandas(pandas.Series.cat.as_unordered)

    def _default_to_pandas(self, op, *args, **kwargs):
        """
        Convert `self` to pandas type and call a pandas cat.`op` on it.

        Parameters
        ----------
        op : str
            Name of pandas function.
        *args : list
            Additional positional arguments to be passed in `op`.
        **kwargs : dict
            Additional keywords arguments to be passed in `op`.

        Returns
        -------
        object
            Result of operation.
        """
        return self._series._default_to_pandas(
            lambda series: op(series.cat, *args, **kwargs)
        )


@_inherit_docstrings(pandas.core.strings.accessor.StringMethods)
class StringMethods(ClassLogger):
    _series: Series
    _query_compiler: BaseQueryCompiler

    def __init__(self, data: Series):
        # Check if dtypes is objects

        self._series = data
        self._query_compiler = data._query_compiler

    @cached_property
    def _Series(self) -> Series:  # noqa: GL08
        # to avoid cyclic import
        from .series import Series

        return Series

    def casefold(self):
        return self._Series(query_compiler=self._query_compiler.str_casefold())

    def cat(self, others=None, sep=None, na_rep=None, join="left"):
        if isinstance(others, self._Series):
            others = others._to_pandas()
        compiler_result = self._query_compiler.str_cat(
            others=others, sep=sep, na_rep=na_rep, join=join
        )
        # if others is None, result is a string. otherwise, it's a series.
        return (
            compiler_result.to_pandas().squeeze()
            if others is None
            else self._Series(query_compiler=compiler_result)
        )

    def decode(self, encoding, errors="strict"):
        return self._Series(
            query_compiler=self._query_compiler.str_decode(encoding, errors)
        )

    def split(self, pat=None, *, n=-1, expand=False, regex=None):
        if expand:
            from .dataframe import DataFrame

            return DataFrame(
                query_compiler=self._query_compiler.str_split(
                    pat=pat, n=n, expand=True, regex=regex
                )
            )
        else:
            return self._Series(
                query_compiler=self._query_compiler.str_split(
                    pat=pat, n=n, expand=expand, regex=regex
                )
            )

    def rsplit(self, pat=None, *, n=-1, expand=False):
        if not pat and pat is not None:
            raise ValueError("rsplit() requires a non-empty pattern match.")

        if expand:
            from .dataframe import DataFrame

            return DataFrame(
                query_compiler=self._query_compiler.str_rsplit(
                    pat=pat, n=n, expand=True
                )
            )
        else:
            return self._Series(
                query_compiler=self._query_compiler.str_rsplit(
                    pat=pat, n=n, expand=expand
                )
            )

    def get(self, i):
        return self._Series(query_compiler=self._query_compiler.str_get(i))

    def join(self, sep):
        if sep is None:
            raise AttributeError("'NoneType' object has no attribute 'join'")
        return self._Series(query_compiler=self._query_compiler.str_join(sep))

    def get_dummies(self, sep="|"):
        return self._Series(query_compiler=self._query_compiler.str_get_dummies(sep))

    def contains(self, pat, case=True, flags=0, na=None, regex=True):
        if pat is None and not case:
            raise AttributeError("'NoneType' object has no attribute 'upper'")
        return self._Series(
            query_compiler=self._query_compiler.str_contains(
                pat, case=case, flags=flags, na=na, regex=regex
            )
        )

    def replace(self, pat, repl, n=-1, case=None, flags=0, regex=False):
        if not (isinstance(repl, str) or callable(repl)):
            raise TypeError("repl must be a string or callable")
        return self._Series(
            query_compiler=self._query_compiler.str_replace(
                pat, repl, n=n, case=case, flags=flags, regex=regex
            )
        )

    def pad(self, width, side="left", fillchar=" "):
        if len(fillchar) != 1:
            raise TypeError("fillchar must be a character, not str")
        return self._Series(
            query_compiler=self._query_compiler.str_pad(
                width, side=side, fillchar=fillchar
            )
        )

    def center(self, width, fillchar=" "):
        if len(fillchar) != 1:
            raise TypeError("fillchar must be a character, not str")
        return self._Series(
            query_compiler=self._query_compiler.str_center(width, fillchar=fillchar)
        )

    def ljust(self, width, fillchar=" "):
        if len(fillchar) != 1:
            raise TypeError("fillchar must be a character, not str")
        return self._Series(
            query_compiler=self._query_compiler.str_ljust(width, fillchar=fillchar)
        )

    def rjust(self, width, fillchar=" "):
        if len(fillchar) != 1:
            raise TypeError("fillchar must be a character, not str")
        return self._Series(
            query_compiler=self._query_compiler.str_rjust(width, fillchar=fillchar)
        )

    def zfill(self, width):
        return self._Series(query_compiler=self._query_compiler.str_zfill(width))

    def wrap(self, width, **kwargs):
        if width <= 0:
            raise ValueError("invalid width {} (must be > 0)".format(width))
        return self._Series(
            query_compiler=self._query_compiler.str_wrap(width, **kwargs)
        )

    def slice(self, start=None, stop=None, step=None):
        if step == 0:
            raise ValueError("slice step cannot be zero")
        return self._Series(
            query_compiler=self._query_compiler.str_slice(
                start=start, stop=stop, step=step
            )
        )

    def slice_replace(self, start=None, stop=None, repl=None):
        return self._Series(
            query_compiler=self._query_compiler.str_slice_replace(
                start=start, stop=stop, repl=repl
            )
        )

    def count(self, pat, flags=0):
        if not isinstance(pat, (str, re.Pattern)):
            raise TypeError("first argument must be string or compiled pattern")
        return self._Series(
            query_compiler=self._query_compiler.str_count(pat, flags=flags)
        )

    def startswith(self, pat, na=None):
        return self._Series(
            query_compiler=self._query_compiler.str_startswith(pat, na=na)
        )

    def encode(self, encoding, errors="strict"):
        return self._Series(
            query_compiler=self._query_compiler.str_encode(encoding, errors)
        )

    def endswith(self, pat, na=None):
        return self._Series(
            query_compiler=self._query_compiler.str_endswith(pat, na=na)
        )

    def findall(self, pat, flags=0):
        if not isinstance(pat, (str, re.Pattern)):
            raise TypeError("first argument must be string or compiled pattern")
        return self._Series(
            query_compiler=self._query_compiler.str_findall(pat, flags=flags)
        )

    def fullmatch(self, pat, case=True, flags=0, na=None):
        if not isinstance(pat, (str, re.Pattern)):
            raise TypeError("first argument must be string or compiled pattern")
        return self._Series(
            query_compiler=self._query_compiler.str_fullmatch(
                pat, case=case, flags=flags, na=na
            )
        )

    def match(self, pat, case=True, flags=0, na=None):
        if not isinstance(pat, (str, re.Pattern)):
            raise TypeError("first argument must be string or compiled pattern")
        return self._Series(
            query_compiler=self._query_compiler.str_match(
                pat, case=case, flags=flags, na=na
            )
        )

    def extract(self, pat, flags=0, expand=True):
        query_compiler = self._query_compiler.str_extract(
            pat, flags=flags, expand=expand
        )
        from .dataframe import DataFrame

        return (
            DataFrame(query_compiler=query_compiler)
            if expand or re.compile(pat).groups > 1
            else self._Series(query_compiler=query_compiler)
        )

    def extractall(self, pat, flags=0):
        return self._Series(
            query_compiler=self._query_compiler.str_extractall(pat, flags)
        )

    def len(self):
        return self._Series(query_compiler=self._query_compiler.str_len())

    def strip(self, to_strip=None):
        return self._Series(
            query_compiler=self._query_compiler.str_strip(to_strip=to_strip)
        )

    def rstrip(self, to_strip=None):
        return self._Series(
            query_compiler=self._query_compiler.str_rstrip(to_strip=to_strip)
        )

    def lstrip(self, to_strip=None):
        return self._Series(
            query_compiler=self._query_compiler.str_lstrip(to_strip=to_strip)
        )

    def partition(self, sep=" ", expand=True):
        if sep is not None and len(sep) == 0:
            raise ValueError("empty separator")

        from .dataframe import DataFrame

        return (DataFrame if expand else self._Series)(
            query_compiler=self._query_compiler.str_partition(sep=sep, expand=expand)
        )

    def removeprefix(self, prefix):
        return self._Series(
            query_compiler=self._query_compiler.str_removeprefix(prefix)
        )

    def removesuffix(self, suffix):
        return self._Series(
            query_compiler=self._query_compiler.str_removesuffix(suffix)
        )

    def repeat(self, repeats):
        return self._Series(query_compiler=self._query_compiler.str_repeat(repeats))

    def rpartition(self, sep=" ", expand=True):
        if sep is not None and len(sep) == 0:
            raise ValueError("empty separator")

        from .dataframe import DataFrame

        return (DataFrame if expand else self._Series)(
            query_compiler=self._query_compiler.str_rpartition(sep=sep, expand=expand)
        )

    def lower(self):
        return self._Series(query_compiler=self._query_compiler.str_lower())

    def upper(self):
        return self._Series(query_compiler=self._query_compiler.str_upper())

    def title(self):
        return self._Series(query_compiler=self._query_compiler.str_title())

    def find(self, sub, start=0, end=None):
        if not isinstance(sub, str):
            raise TypeError(
                "expected a string object, not {0}".format(type(sub).__name__)
            )
        return self._Series(
            query_compiler=self._query_compiler.str_find(sub, start=start, end=end)
        )

    def rfind(self, sub, start=0, end=None):
        if not isinstance(sub, str):
            raise TypeError(
                "expected a string object, not {0}".format(type(sub).__name__)
            )
        return self._Series(
            query_compiler=self._query_compiler.str_rfind(sub, start=start, end=end)
        )

    def index(self, sub, start=0, end=None):
        if not isinstance(sub, str):
            raise TypeError(
                "expected a string object, not {0}".format(type(sub).__name__)
            )
        return self._Series(
            query_compiler=self._query_compiler.str_index(sub, start=start, end=end)
        )

    def rindex(self, sub, start=0, end=None):
        if not isinstance(sub, str):
            raise TypeError(
                "expected a string object, not {0}".format(type(sub).__name__)
            )
        return self._Series(
            query_compiler=self._query_compiler.str_rindex(sub, start=start, end=end)
        )

    def capitalize(self):
        return self._Series(query_compiler=self._query_compiler.str_capitalize())

    def swapcase(self):
        return self._Series(query_compiler=self._query_compiler.str_swapcase())

    def normalize(self, form):
        return self._Series(query_compiler=self._query_compiler.str_normalize(form))

    def translate(self, table):
        return self._Series(query_compiler=self._query_compiler.str_translate(table))

    def isalnum(self):
        return self._Series(query_compiler=self._query_compiler.str_isalnum())

    def isalpha(self):
        return self._Series(query_compiler=self._query_compiler.str_isalpha())

    def isdigit(self):
        return self._Series(query_compiler=self._query_compiler.str_isdigit())

    def isspace(self):
        return self._Series(query_compiler=self._query_compiler.str_isspace())

    def islower(self):
        return self._Series(query_compiler=self._query_compiler.str_islower())

    def isupper(self):
        return self._Series(query_compiler=self._query_compiler.str_isupper())

    def istitle(self):
        return self._Series(query_compiler=self._query_compiler.str_istitle())

    def isnumeric(self):
        return self._Series(query_compiler=self._query_compiler.str_isnumeric())

    def isdecimal(self):
        return self._Series(query_compiler=self._query_compiler.str_isdecimal())

    def __getitem__(self, key):  # noqa: GL08
        return self._Series(query_compiler=self._query_compiler.str___getitem__(key))

    def _default_to_pandas(self, op, *args, **kwargs):
        """
        Convert `self` to pandas type and call a pandas str.`op` on it.

        Parameters
        ----------
        op : str
            Name of pandas function.
        *args : list
            Additional positional arguments to be passed in `op`.
        **kwargs : dict
            Additional keywords arguments to be passed in `op`.

        Returns
        -------
        object
            Result of operation.
        """
        return self._series._default_to_pandas(
            lambda series: op(series.str, *args, **kwargs)
        )


@_inherit_docstrings(pandas.core.indexes.accessors.CombinedDatetimelikeProperties)
class DatetimeProperties(ClassLogger):  # noqa: GL08
    _series: Series
    _query_compiler: BaseQueryCompiler

    def __init__(self, data: Series):
        self._series = data
        self._query_compiler = data._query_compiler

    @cached_property
    def _Series(self) -> Series:  # noqa: GL08
        # to avoid cyclic import
        from .series import Series

        return Series

    @property
    def date(self):
        return self._Series(query_compiler=self._query_compiler.dt_date())

    @property
    def time(self):
        return self._Series(query_compiler=self._query_compiler.dt_time())

    @property
    def timetz(self):
        return self._Series(query_compiler=self._query_compiler.dt_timetz())

    @property
    def year(self):
        return self._Series(query_compiler=self._query_compiler.dt_year())

    @property
    def month(self):
        return self._Series(query_compiler=self._query_compiler.dt_month())

    @property
    def day(self):
        return self._Series(query_compiler=self._query_compiler.dt_day())

    @property
    def hour(self):
        return self._Series(query_compiler=self._query_compiler.dt_hour())

    @property
    def minute(self):
        return self._Series(query_compiler=self._query_compiler.dt_minute())

    @property
    def second(self):
        return self._Series(query_compiler=self._query_compiler.dt_second())

    @property
    def microsecond(self):
        return self._Series(query_compiler=self._query_compiler.dt_microsecond())

    @property
    def nanosecond(self):
        return self._Series(query_compiler=self._query_compiler.dt_nanosecond())

    @property
    def dayofweek(self):
        return self._Series(query_compiler=self._query_compiler.dt_dayofweek())

    day_of_week = dayofweek

    @property
    def weekday(self):
        return self._Series(query_compiler=self._query_compiler.dt_weekday())

    @property
    def dayofyear(self):
        return self._Series(query_compiler=self._query_compiler.dt_dayofyear())

    day_of_year = dayofyear

    @property
    def quarter(self):
        return self._Series(query_compiler=self._query_compiler.dt_quarter())

    @property
    def is_month_start(self):
        return self._Series(query_compiler=self._query_compiler.dt_is_month_start())

    @property
    def is_month_end(self):
        return self._Series(query_compiler=self._query_compiler.dt_is_month_end())

    @property
    def is_quarter_start(self):
        return self._Series(query_compiler=self._query_compiler.dt_is_quarter_start())

    @property
    def is_quarter_end(self):
        return self._Series(query_compiler=self._query_compiler.dt_is_quarter_end())

    @property
    def is_year_start(self):
        return self._Series(query_compiler=self._query_compiler.dt_is_year_start())

    @property
    def is_year_end(self):
        return self._Series(query_compiler=self._query_compiler.dt_is_year_end())

    @property
    def is_leap_year(self):
        return self._Series(query_compiler=self._query_compiler.dt_is_leap_year())

    @property
    def daysinmonth(self):
        return self._Series(query_compiler=self._query_compiler.dt_daysinmonth())

    @property
    def days_in_month(self):
        return self._Series(query_compiler=self._query_compiler.dt_days_in_month())

    @property
    def tz(self) -> "tzinfo | None":
        dtype = self._series.dtype
        if isinstance(dtype, np.dtype):
            return None
        return dtype.tz

    @property
    def freq(self):  # noqa: GL08
        return self._query_compiler.dt_freq().to_pandas().squeeze()

    @property
    def unit(self):  # noqa: GL08
        # use `iloc[0]` to return scalar
        return self._Series(query_compiler=self._query_compiler.dt_unit()).iloc[0]

    def as_unit(self, *args, **kwargs):  # noqa: GL08
        return self._Series(
            query_compiler=self._query_compiler.dt_as_unit(*args, **kwargs)
        )

    def to_period(self, *args, **kwargs):
        return self._Series(
            query_compiler=self._query_compiler.dt_to_period(*args, **kwargs)
        )

    def asfreq(self, *args, **kwargs):
        return self._Series(
            query_compiler=self._query_compiler.dt_asfreq(*args, **kwargs)
        )

    def to_pydatetime(self):
        return self._Series(
            query_compiler=self._query_compiler.dt_to_pydatetime()
        ).to_numpy()

    def tz_localize(self, *args, **kwargs):
        return self._Series(
            query_compiler=self._query_compiler.dt_tz_localize(*args, **kwargs)
        )

    def tz_convert(self, *args, **kwargs):
        return self._Series(
            query_compiler=self._query_compiler.dt_tz_convert(*args, **kwargs)
        )

    def normalize(self, *args, **kwargs):
        return self._Series(
            query_compiler=self._query_compiler.dt_normalize(*args, **kwargs)
        )

    def strftime(self, *args, **kwargs):
        return self._Series(
            query_compiler=self._query_compiler.dt_strftime(*args, **kwargs)
        )

    def round(self, *args, **kwargs):
        return self._Series(
            query_compiler=self._query_compiler.dt_round(*args, **kwargs)
        )

    def floor(self, *args, **kwargs):
        return self._Series(
            query_compiler=self._query_compiler.dt_floor(*args, **kwargs)
        )

    def ceil(self, *args, **kwargs):
        return self._Series(
            query_compiler=self._query_compiler.dt_ceil(*args, **kwargs)
        )

    def month_name(self, *args, **kwargs):
        return self._Series(
            query_compiler=self._query_compiler.dt_month_name(*args, **kwargs)
        )

    def day_name(self, *args, **kwargs):
        return self._Series(
            query_compiler=self._query_compiler.dt_day_name(*args, **kwargs)
        )

    def total_seconds(self, *args, **kwargs):
        return self._Series(
            query_compiler=self._query_compiler.dt_total_seconds(*args, **kwargs)
        )

    def to_pytimedelta(self) -> "npt.NDArray[np.object_]":
        res = self._query_compiler.dt_to_pytimedelta()
        return res.to_numpy()[:, 0]

    @property
    def seconds(self):
        return self._Series(query_compiler=self._query_compiler.dt_seconds())

    @property
    def days(self):
        return self._Series(query_compiler=self._query_compiler.dt_days())

    @property
    def microseconds(self):
        return self._Series(query_compiler=self._query_compiler.dt_microseconds())

    @property
    def nanoseconds(self):
        return self._Series(query_compiler=self._query_compiler.dt_nanoseconds())

    @property
    def components(self):
        from .dataframe import DataFrame

        return DataFrame(query_compiler=self._query_compiler.dt_components())

    def isocalendar(self):
        from .dataframe import DataFrame

        return DataFrame(query_compiler=self._query_compiler.dt_isocalendar())

    @property
    def qyear(self):  # noqa: GL08
        return self._Series(query_compiler=self._query_compiler.dt_qyear())

    @property
    def start_time(self):
        return self._Series(query_compiler=self._query_compiler.dt_start_time())

    @property
    def end_time(self):
        return self._Series(query_compiler=self._query_compiler.dt_end_time())

    def to_timestamp(self, *args, **kwargs):
        return self._Series(
            query_compiler=self._query_compiler.dt_to_timestamp(*args, **kwargs)
        )
