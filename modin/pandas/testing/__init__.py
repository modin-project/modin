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
Public testing utility functions.
"""

from __future__ import annotations

from typing import Literal

from pandas._libs import lib
from pandas.testing import assert_extension_array_equal
from pandas.testing import assert_frame_equal as pd_assert_frame_equal
from pandas.testing import assert_index_equal
from pandas.testing import assert_series_equal as pd_assert_series_equal

from modin.utils import _inherit_docstrings, try_cast_to_pandas


@_inherit_docstrings(pd_assert_frame_equal, apilink="pandas.testing.assert_frame_equal")
def assert_frame_equal(
    left,
    right,
    check_dtype: bool | Literal["equiv"] = True,
    check_index_type: bool | Literal["equiv"] = "equiv",
    check_column_type: bool | Literal["equiv"] = "equiv",
    check_frame_type: bool = True,
    check_names: bool = True,
    by_blocks: bool = False,
    check_exact: bool | lib.NoDefault = lib.no_default,
    check_datetimelike_compat: bool = False,
    check_categorical: bool = True,
    check_like: bool = False,
    check_freq: bool = True,
    check_flags: bool = True,
    rtol: float | lib.NoDefault = lib.no_default,
    atol: float | lib.NoDefault = lib.no_default,
    obj: str = "DataFrame",
) -> None:
    left = try_cast_to_pandas(left)
    right = try_cast_to_pandas(right)
    pd_assert_frame_equal(
        left,
        right,
        check_dtype=check_dtype,
        check_index_type=check_index_type,
        check_column_type=check_column_type,
        check_frame_type=check_frame_type,
        check_names=check_names,
        by_blocks=by_blocks,
        check_exact=check_exact,
        check_datetimelike_compat=check_datetimelike_compat,
        check_categorical=check_categorical,
        check_like=check_like,
        check_freq=check_freq,
        check_flags=check_flags,
        rtol=rtol,
        atol=atol,
        obj=obj,
    )


@_inherit_docstrings(
    pd_assert_series_equal, apilink="pandas.testing.assert_series_equal"
)
def assert_series_equal(
    left,
    right,
    check_dtype: bool | Literal["equiv"] = True,
    check_index_type: bool | Literal["equiv"] = "equiv",
    check_series_type: bool = True,
    check_names: bool = True,
    check_exact: bool | lib.NoDefault = lib.no_default,
    check_datetimelike_compat: bool = False,
    check_categorical: bool = True,
    check_category_order: bool = True,
    check_freq: bool = True,
    check_flags: bool = True,
    rtol: float | lib.NoDefault = lib.no_default,
    atol: float | lib.NoDefault = lib.no_default,
    obj: str = "Series",
    *,
    check_index: bool = True,
    check_like: bool = False,
) -> None:
    left = try_cast_to_pandas(left)
    right = try_cast_to_pandas(right)
    pd_assert_series_equal(
        left,
        right,
        check_dtype=check_dtype,
        check_index_type=check_index_type,
        check_series_type=check_series_type,
        check_names=check_names,
        check_exact=check_exact,
        check_datetimelike_compat=check_datetimelike_compat,
        check_categorical=check_categorical,
        check_category_order=check_category_order,
        check_freq=check_freq,
        check_flags=check_flags,
        rtol=rtol,
        atol=atol,
        obj=obj,
        check_index=check_index,
        check_like=check_like,
    )


__all__ = [
    "assert_extension_array_equal",
    "assert_frame_equal",
    "assert_series_equal",
    "assert_index_equal",
]
