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
    # breakpoint()
    df_equals(md_res, pd_res)

    index_loc = get_loc(pd_item2, loc)

    pd_res = get_reference(pd_item2, loc=index_loc, value=pd_item1)
    md_res = md_item2._query_compiler.insert_item(
        axis=axis, loc=index_loc, value=md_item1._query_compiler, replace=replace
    ).to_pandas()

    df_equals(md_res, pd_res)
