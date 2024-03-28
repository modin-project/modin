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
from modin.config import NPartitions
from modin.tests.pandas.utils import create_test_dfs, df_equals, test_data_values

NPartitions.put(4)


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("item_length", [0, 1, 2])
@pytest.mark.parametrize("loc", ["first", "first + 1", "middle", "penult", "last"])
@pytest.mark.parametrize("replace", [True, False])
def test_insert_item(axis, item_length, loc, replace):
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
            if replace:
                loc += 1
            second_mask = df.iloc[loc:]
        else:
            first_mask = df.iloc[:, :loc]
            if replace:
                loc += 1
            second_mask = df.iloc[:, loc:]
        return pandas.concat([first_mask, value, second_mask], axis=axis)

    md_frames, pd_frames = create_test_dfs(data, post_fn=post_fn)
    md_item1, md_item2 = md_frames
    pd_item1, pd_item2 = pd_frames

    index_loc = get_loc(pd_item1, loc)

    pd_res = get_reference(pd_item1, loc=index_loc, value=pd_item2)
    md_res = md_item1._query_compiler.insert_item(
        axis=axis, loc=index_loc, value=md_item2._query_compiler, replace=replace
    ).to_pandas()
    df_equals(
        md_res,
        pd_res,
        # This test causes an empty slice to be generated thus triggering:
        # https://github.com/modin-project/modin/issues/5974
        check_dtypes=axis != 0,
    )

    index_loc = get_loc(pd_item2, loc)

    pd_res = get_reference(pd_item2, loc=index_loc, value=pd_item1)
    md_res = md_item2._query_compiler.insert_item(
        axis=axis, loc=index_loc, value=md_item1._query_compiler, replace=replace
    ).to_pandas()

    df_equals(
        md_res,
        pd_res,
        # This test causes an empty slice to be generated thus triggering:
        # https://github.com/modin-project/modin/issues/5974
        check_dtypes=axis != 0,
    )


@pytest.mark.parametrize("num_rows", list(range(1, 5)), ids=lambda x: f"num_rows={x}")
@pytest.mark.parametrize("num_cols", list(range(1, 5)), ids=lambda x: f"num_cols={x}")
def test_repr_size_issue_6104(num_rows, num_cols):
    # this tests an edge case where we used to select exactly num_cols / 2 + 1 columns
    # from both the front and the back of the dataframe, but the dataframe is such a
    # length that the front and back columns overlap at one column. The result is that
    # we convert one column twice to pandas, although we would never see the duplicate
    # column in the output because pandas would also only represent the num_cols / 2
    # columns from the front and back.
    df = pd.DataFrame([list(range(4)) for _ in range(4)])
    pandas_repr_df = df._build_repr_df(num_rows, num_cols)
    assert pandas_repr_df.columns.is_unique
    assert pandas_repr_df.index.is_unique
