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

from itertools import product

import matplotlib
import numpy as np
import pandas
import pytest

import modin.pandas as pd
from modin.config import NativeDataframeMode, NPartitions
from modin.tests.pandas.utils import create_test_dfs, df_equals

NPartitions.put(4)

# Force matplotlib to not use any Xwindows backend.
matplotlib.use("Agg")


@pytest.mark.parametrize(
    "data_frame_mode_pair", list(product(NativeDataframeMode.choices, repeat=2))
)
def test_fillna_4660(data_frame_mode_pair):
    modin_df_1, pandas_df_1 = create_test_dfs(
        {"a": ["a"], "b": ["b"], "c": [pd.NA]},
        index=["row1"],
        data_frame_mode=data_frame_mode_pair[0],
    )
    modin_df_2, pandas_df_2 = create_test_dfs(
        {"a": ["a"], "b": ["b"], "c": [pd.NA]},
        index=["row1"],
        data_frame_mode=data_frame_mode_pair[1],
    )
    modin_result = modin_df_1["c"].fillna(modin_df_2["b"])
    pandas_result = pandas_df_1["c"].fillna(pandas_df_2["b"])
    df_equals(modin_result, pandas_result)


@pytest.mark.parametrize(
    "data_frame_mode_pair", list(product(NativeDataframeMode.choices, repeat=2))
)
def test_fillna_dict_series(data_frame_mode_pair):
    frame_data = {
        "a": [np.nan, 1, 2, np.nan, np.nan],
        "b": [1, 2, 3, np.nan, np.nan],
        "c": [np.nan, 1, 2, 3, 4],
    }
    df = pandas.DataFrame(frame_data)
    modin_df = pd.DataFrame(frame_data)
    modin_df_1, pandas_df_1 = create_test_dfs(
        frame_data, data_frame_mode=data_frame_mode_pair[0]
    )
    modin_df_2, pandas_df_2 = create_test_dfs(
        frame_data, data_frame_mode=data_frame_mode_pair[1]
    )

    df_equals(modin_df.fillna({"a": 0, "b": 5}), df.fillna({"a": 0, "b": 5}))

    df_equals(
        modin_df.fillna({"a": 0, "b": 5, "d": 7}),
        df.fillna({"a": 0, "b": 5, "d": 7}),
    )

    # Series treated same as dict
    df_equals(
        modin_df_1.fillna(modin_df_2.max()), pandas_df_1.fillna(pandas_df_2.max())
    )


@pytest.mark.parametrize(
    "data_frame_mode_pair", list(product(NativeDataframeMode.choices, repeat=2))
)
def test_fillna_dataframe(data_frame_mode_pair):
    frame_data = {
        "a": [np.nan, 1, 2, np.nan, np.nan],
        "b": [1, 2, 3, np.nan, np.nan],
        "c": [np.nan, 1, 2, 3, 4],
    }
    modin_df_1, pandas_df_1 = create_test_dfs(
        frame_data, index=list("VWXYZ"), data_frame_mode=data_frame_mode_pair[0]
    )
    modin_df_2, pandas_df_2 = create_test_dfs(
        {"a": [np.nan, 10, 20, 30, 40], "b": [50, 60, 70, 80, 90], "foo": ["bar"] * 5},
        index=list("VWXuZ"),
        data_frame_mode=data_frame_mode_pair[1],
    )

    # only those columns and indices which are shared get filled
    df_equals(modin_df_1.fillna(modin_df_2), pandas_df_1.fillna(pandas_df_2))
