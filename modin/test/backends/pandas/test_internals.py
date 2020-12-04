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

import modin.pandas as pd

import pandas
import pytest

from modin.pandas.test.utils import (
    test_data_values,
    create_test_dfs,
    df_equals,
)

pd.DEFAULT_NPARTITIONS = 4


def test_aligning_blocks():
    # Test problem when modin frames have the same number of rows, but different
    # blocks (partition.list_of_blocks). See #2322 for details
    accm = pd.DataFrame(["-22\n"] * 162)
    accm = accm.iloc[2:, :]
    accm.reset_index(drop=True, inplace=True)
    accm["T"] = pd.Series(["24.67\n"] * 145)

    # see #2322 for details
    repr(accm)


def test_aligning_blocks_with_duplicated_index():
    # Same problem as in `test_aligning_blocks` but with duplicated values in index.
    data11 = [0, 1]
    data12 = [2, 3]

    data21 = [0]
    data22 = [1, 2, 3]

    df1 = pd.DataFrame(data11).append(pd.DataFrame(data12))
    df2 = pd.DataFrame(data21).append(pd.DataFrame(data22))

    repr(df1 - df2)


def test_aligning_partitions():
    data = [0, 1, 2, 3, 4, 5]
    modin_df1, _ = create_test_dfs({"a": data, "b": data})
    modin_df = modin_df1.loc[:2]

    modin_df2 = modin_df.append(modin_df)

    modin_df2["c"] = modin_df1["b"]
    repr(modin_df2)


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("item_length", [0, 1, 2])
@pytest.mark.parametrize("loc", ["first", "first + 1", "middle", "penult", "last"])
def test_insert_item(axis, item_length, loc):
    data = test_data_values[0]

    def post_fn(df):
        return (
            (df.iloc[:, :-item_length], df.iloc[:, -item_length:])
            if axis
            else (df.iloc[:-item_length, :], df.iloc[-item_length:, :])
        )

    def get_loc(frame, loc):
        locs_dict = {
            "first": 0,
            "first + 1": 1,
            "middle": len(frame.axes[axis]) // 2,
            "penult": len(frame.axes[axis]) - 1,
            "last": len(frame.axes[axis]),
        }
        return locs_dict[loc]

    def get_reference(df, value, loc):
        if axis == 0:
            first_mask = df.iloc[:loc]
            second_mask = df.iloc[loc:]
        else:
            first_mask = df.iloc[:, :loc]
            second_mask = df.iloc[:, loc:]
        return pandas.concat([first_mask, value, second_mask], axis=axis)

    md_frames, pd_frames = create_test_dfs(data, post_fn=post_fn)
    md_item1, md_item2 = md_frames
    pd_item1, pd_item2 = pd_frames

    index_loc = get_loc(pd_item1, loc)

    pd_res = get_reference(pd_item1, loc=index_loc, value=pd_item2)
    md_res = md_item1._query_compiler.insert_item(
        axis=axis, loc=index_loc, value=md_item2._query_compiler
    ).to_pandas()

    df_equals(md_res, pd_res)

    index_loc = get_loc(pd_item2, loc)

    pd_res = get_reference(pd_item2, loc=index_loc, value=pd_item1)
    md_res = md_item2._query_compiler.insert_item(
        axis=axis, loc=index_loc, value=md_item1._query_compiler
    ).to_pandas()

    df_equals(md_res, pd_res)
