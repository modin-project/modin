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

import pandas
import pytest

import modin.pandas as pd
from modin.tests.pandas.native_df_interoperability.utils import (
    create_test_df_in_defined_mode,
    create_test_series_in_defined_mode,
)
from modin.tests.pandas.utils import default_to_pandas_ignore_string, df_equals

# Our configuration in pytest.ini requires that we explicitly catch all
# instances of defaulting to pandas, but some test modules, like this one,
# have too many such instances.
pytestmark = pytest.mark.filterwarnings(default_to_pandas_ignore_string)


def test_cut(df_mode_pair):
    modin_x, pandas_x = create_test_series_in_defined_mode(
        [1, 3], native=df_mode_pair[0]
    )
    modin_bins, pandas_bins = create_test_series_in_defined_mode(
        [0, 2], native=df_mode_pair[1]
    )

    def operation(*, lib, x, bins):
        return lib.cut(x, bins)

    df_equals(
        operation(lib=pd, x=modin_x, bins=modin_bins),
        operation(lib=pandas, x=pandas_x, bins=pandas_bins),
    )


def test_qcut(df_mode_pair):
    modin_x, pandas_x = create_test_series_in_defined_mode(
        [1, 2, 3, 4], native=df_mode_pair[0]
    )
    modin_quantiles, pandas_quantiles = create_test_series_in_defined_mode(
        [0, 0.5, 1], native=df_mode_pair[1]
    )

    def operation(*, lib, x, quantiles):
        return lib.qcut(x, quantiles)

    df_equals(
        operation(lib=pd, x=modin_x, quantiles=modin_quantiles),
        operation(lib=pandas, x=pandas_x, quantiles=pandas_quantiles),
    )


def test_merge_ordered(df_mode_pair):
    modin_left, pandas_left = create_test_df_in_defined_mode(
        {
            "key": ["a", "c", "e", "a", "c", "e"],
            "lvalue": [1, 2, 3, 1, 2, 3],
            "group": ["a", "a", "a", "b", "b", "b"],
        },
        native=df_mode_pair[0],
    )
    modin_right, pandas_right = create_test_df_in_defined_mode(
        {"key": ["b", "c", "d"], "rvalue": [1, 2, 3]},
        native=df_mode_pair[1],
    )

    def operation(*, lib, left, right):
        return lib.merge_ordered(left, right, fill_method="ffill", left_by="group")

    df_equals(
        operation(lib=pd, left=modin_left, right=modin_right),
        operation(lib=pandas, left=pandas_left, right=pandas_right),
    )


def test_merge_asof(df_mode_pair):
    modin_left, pandas_left = create_test_df_in_defined_mode(
        {"a": [1, 5, 10], "left_val": ["a", "b", "c"]}, native=df_mode_pair[0]
    )
    modin_right, pandas_right = create_test_df_in_defined_mode(
        {"a": [1, 2, 3, 6, 7], "right_val": [1, 2, 3, 6, 7]},
        native=df_mode_pair[1],
    )

    def operation(*, lib, left, right):
        return lib.merge_asof(left, right, on="a")

    df_equals(
        operation(lib=pd, left=modin_left, right=modin_right),
        operation(lib=pandas, left=pandas_left, right=pandas_right),
    )
