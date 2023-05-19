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

import pytest
import numpy as np
import pandas
import modin.pandas as pd

from .utils import (
    df_equals,
    test_data,
    test_data_values,
    test_data_keys,
    eval_general,
    create_test_dfs,
)
from modin.test.test_utils import warns_that_defaulting_to_pandas
from modin.config import NPartitions

NPartitions.put(4)


def create_test_series(vals):
    if isinstance(vals, dict):
        modin_series = pd.Series(vals[next(iter(vals.keys()))])
        pandas_series = pandas.Series(vals[next(iter(vals.keys()))])
    else:
        modin_series = pd.Series(vals)
        pandas_series = pandas.Series(vals)
    return modin_series, pandas_series


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("min_periods", [None, 5])
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize(
    "method, kwargs",
    [
        ("count", {}),
        ("sum", {}),
        ("mean", {}),
        ("median", {}),
        ("skew", {}),
        ("kurt", {}),
        ("var", {"ddof": 0}),
        ("std", {"ddof": 0}),
        ("min", {}),
        ("max", {}),
        ("rank", {}),
        ("sem", {"ddof": 0}),
        ("quantile", {"q": 0.1}),
    ],
)
def test_dataframe(data, min_periods, axis, method, kwargs):
    eval_general(
        *create_test_dfs(data),
        lambda df: getattr(
            df.expanding(min_periods=min_periods, center=True, axis=axis), method
        )(**kwargs)
    )


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("min_periods", [None, 5])
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("method", ["corr", "cov"])
def test_dataframe_corr_cov(data, min_periods, axis, method):
    with warns_that_defaulting_to_pandas():
        eval_general(
            *create_test_dfs(data),
            lambda df: getattr(
                df.expanding(min_periods=min_periods, center=True, axis=axis), method
            )()
        )


@pytest.mark.parametrize("method", ["corr", "cov"])
def test_dataframe_corr_cov_with_self(method):
    mdf, pdf = create_test_dfs(test_data["float_nan_data"])
    with warns_that_defaulting_to_pandas():
        eval_general(
            mdf,
            pdf,
            lambda df, other: getattr(df.expanding(), method)(other=other),
            other=pdf,
            md_extra_kwargs={"other": mdf},
        )


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("min_periods", [None, 5])
def test_dataframe_agg(data, min_periods):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)
    pandas_expanded = pandas_df.expanding(
        min_periods=min_periods,
        center=True,
        axis=0,
    )
    modin_expanded = modin_df.expanding(
        min_periods=min_periods,
        center=True,
        axis=0,
    )
    # aggregates are only supported on axis 0
    df_equals(modin_expanded.aggregate(np.sum), pandas_expanded.aggregate(np.sum))
    df_equals(
        pandas_expanded.aggregate([np.sum, np.mean]),
        modin_expanded.aggregate([np.sum, np.mean]),
    )


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("min_periods", [None, 5])
@pytest.mark.parametrize(
    "method, kwargs",
    [
        ("count", {}),
        ("sum", {}),
        ("mean", {}),
        ("median", {}),
        ("skew", {}),
        ("kurt", {}),
        ("corr", {}),
        ("cov", {}),
        ("var", {"ddof": 0}),
        ("std", {"ddof": 0}),
        ("min", {}),
        ("max", {}),
        ("rank", {}),
        ("sem", {"ddof": 0}),
        ("quantile", {"q": 0.1}),
    ],
)
def test_series(data, min_periods, method, kwargs):
    eval_general(
        *create_test_series(data),
        lambda df: getattr(df.expanding(min_periods=min_periods, center=True), method)(
            **kwargs
        )
    )


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("min_periods", [None, 5])
def test_series_agg(data, min_periods):
    modin_series, pandas_series = create_test_series(data)
    pandas_expanded = pandas_series.expanding(
        min_periods=min_periods,
        center=True,
    )
    modin_expanded = modin_series.expanding(
        min_periods=min_periods,
        center=True,
    )

    df_equals(modin_expanded.aggregate(np.sum), pandas_expanded.aggregate(np.sum))
    df_equals(
        pandas_expanded.aggregate([np.sum, np.mean]),
        modin_expanded.aggregate([np.sum, np.mean]),
    )


@pytest.mark.parametrize("method", ["corr", "cov"])
def test_series_corr_cov_with_self(method):
    mdf, pdf = create_test_series(test_data["float_nan_data"])
    eval_general(
        mdf,
        pdf,
        lambda df, other: getattr(df.expanding(), method)(other=other),
        other=pdf,
        md_extra_kwargs={"other": mdf},
    )
