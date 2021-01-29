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

from modin.pandas.test.utils import (
    df_equals,
)


def compare_reindex(
    md_df,
    pd_df,
    md_labels=None,
    pd_labels=None,
    fill_value=None,
    axis=None,
    level=None,
    index=None,
    columns=None,
):
    if axis is not None and (index is not None or columns is not None):
        raise TypeError("cannot specify both axis and either index or columns")
    if axis is not None:
        pandas_result = pd_df.reindex(
            pd_labels, fill_value=fill_value, axis=axis, level=level
        )
        with pytest.warns(UserWarning):
            modin_result = md_df.reindex(
                md_labels, fill_value=fill_value, axis=axis, level=level
            )
    else:
        pandas_result = pd_df.reindex(
            pd_labels, fill_value=fill_value, level=level, index=index, columns=columns
        )
        with pytest.warns(UserWarning):
            modin_result = md_df.reindex(
                md_labels,
                fill_value=fill_value,
                level=level,
                index=index,
                columns=columns,
            )
    df_equals(modin_result, pandas_result)


def test_multiindex_default_to_pandas():
    data1, data2 = np.random.randint(1, 20, (5, 5)), np.random.randint(10, 25, 6)
    index = np.array(["AUD", "BRL", "CAD", "EUR", "INR"])
    modin_midx = pd.MultiIndex.from_product(
        [["Bank_1", "Bank_2"], ["AUD", "CAD", "EUR"]], names=["Bank", "Curency"]
    )
    pandas_midx = pandas.MultiIndex.from_product(
        [["Bank_1", "Bank_2"], ["AUD", "CAD", "EUR"]], names=["Bank", "Curency"]
    )
    modin_df1, modin_df2 = (
        pd.DataFrame(data=data1, index=index, columns=index),
        pd.DataFrame(data2, modin_midx),
    )
    pandas_df1, pandas_df2 = (
        pandas.DataFrame(data=data1, index=index, columns=index),
        pandas.DataFrame(data2, pandas_midx),
    )
    modin_df2.columns, pandas_df2.columns = ["Notional"], ["Notional"]
    md_labels = pd.MultiIndex.from_product([modin_df2.index.levels[0], modin_df1.index])
    pd_labels = pandas.MultiIndex.from_product(
        [pandas_df2.index.levels[0], pandas_df1.index]
    )
    # reindex without axis, index, or columns
    compare_reindex(
        md_df=modin_df2,
        pd_df=pandas_df2,
        md_labels=md_labels,
        pd_labels=pd_labels,
        fill_value=0,
    )
    # reindex with only axis
    compare_reindex(
        md_df=modin_df2,
        pd_df=pandas_df2,
        md_labels=md_labels,
        pd_labels=pd_labels,
        fill_value=0,
        axis=0,
    )
    # reindex with axis and level
    compare_reindex(
        md_df=modin_df2,
        pd_df=pandas_df2,
        md_labels=md_labels,
        pd_labels=pd_labels,
        fill_value=0,
        axis=1,
        level=0,
    )
