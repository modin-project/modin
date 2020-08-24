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
import pandas
import matplotlib
import modin.pandas as pd

from modin.pandas.test.utils import (
    df_equals,
    arg_keys,
    test_data_values,
    test_data_keys,
    axis_keys,
    axis_values,
    bool_arg_keys,
    bool_arg_values,
)

pd.DEFAULT_NPARTITIONS = 4

# Force matplotlib to not use any Xwindows backend.
matplotlib.use("Agg")


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize(
    "skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys)
)
@pytest.mark.parametrize(
    "bool_only", bool_arg_values, ids=arg_keys("bool_only", bool_arg_keys)
)
def test_all(data, axis, skipna, bool_only):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    try:
        pandas_result = pandas_df.all(axis=axis, skipna=skipna, bool_only=bool_only)
    except Exception as e:
        with pytest.raises(type(e)):
            modin_df.all(axis=axis, skipna=skipna, bool_only=bool_only)
    else:
        modin_result = modin_df.all(axis=axis, skipna=skipna, bool_only=bool_only)
        df_equals(modin_result, pandas_result)

    # Test when axis is None. This will get repeated but easier than using list in parameterize decorator
    try:
        pandas_result = pandas_df.all(axis=None, skipna=skipna, bool_only=bool_only)
    except Exception as e:
        with pytest.raises(type(e)):
            modin_df.all(axis=None, skipna=skipna, bool_only=bool_only)
    else:
        modin_result = modin_df.all(axis=None, skipna=skipna, bool_only=bool_only)
        df_equals(modin_result, pandas_result)

    try:
        pandas_result = pandas_df.T.all(axis=axis, skipna=skipna, bool_only=bool_only)
    except Exception as e:
        with pytest.raises(type(e)):
            modin_df.T.all(axis=axis, skipna=skipna, bool_only=bool_only)
    else:
        modin_result = modin_df.T.all(axis=axis, skipna=skipna, bool_only=bool_only)
        df_equals(modin_result, pandas_result)

    # Test when axis is None. This will get repeated but easier than using list in parameterize decorator
    try:
        pandas_result = pandas_df.T.all(axis=None, skipna=skipna, bool_only=bool_only)
    except Exception as e:
        with pytest.raises(type(e)):
            modin_df.T.all(axis=None, skipna=skipna, bool_only=bool_only)
    else:
        modin_result = modin_df.T.all(axis=None, skipna=skipna, bool_only=bool_only)
        df_equals(modin_result, pandas_result)

    # test level
    modin_df_multi_level = modin_df.copy()
    pandas_df_multi_level = pandas_df.copy()
    axis = modin_df._get_axis_number(axis) if axis is not None else 0
    levels = 3
    axis_names_list = [["a", "b", "c"], None]
    for axis_names in axis_names_list:
        if axis == 0:
            new_idx = pandas.MultiIndex.from_tuples(
                [(i // 4, i // 2, i) for i in range(len(modin_df.index))],
                names=axis_names,
            )
            modin_df_multi_level.index = new_idx
            pandas_df_multi_level.index = new_idx
        else:
            new_col = pandas.MultiIndex.from_tuples(
                [(i // 4, i // 2, i) for i in range(len(modin_df.columns))],
                names=axis_names,
            )
            modin_df_multi_level.columns = new_col
            pandas_df_multi_level.columns = new_col

        for level in list(range(levels)) + (axis_names if axis_names else []):
            try:
                pandas_multi_level_result = pandas_df_multi_level.all(
                    axis=axis, bool_only=bool_only, level=level, skipna=skipna
                )

            except Exception as e:
                with pytest.raises(type(e)):
                    modin_df_multi_level.all(
                        axis=axis, bool_only=bool_only, level=level, skipna=skipna
                    )
            else:
                modin_multi_level_result = modin_df_multi_level.all(
                    axis=axis, bool_only=bool_only, level=level, skipna=skipna
                )

                df_equals(modin_multi_level_result, pandas_multi_level_result)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize(
    "skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys)
)
@pytest.mark.parametrize(
    "bool_only", bool_arg_values, ids=arg_keys("bool_only", bool_arg_keys)
)
def test_any(data, axis, skipna, bool_only):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    try:
        pandas_result = pandas_df.any(axis=axis, skipna=skipna, bool_only=bool_only)
    except Exception as e:
        with pytest.raises(type(e)):
            modin_df.any(axis=axis, skipna=skipna, bool_only=bool_only)
    else:
        modin_result = modin_df.any(axis=axis, skipna=skipna, bool_only=bool_only)
        df_equals(modin_result, pandas_result)

    try:
        pandas_result = pandas_df.any(axis=None, skipna=skipna, bool_only=bool_only)
    except Exception as e:
        with pytest.raises(type(e)):
            modin_df.any(axis=None, skipna=skipna, bool_only=bool_only)
    else:
        modin_result = modin_df.any(axis=None, skipna=skipna, bool_only=bool_only)
        df_equals(modin_result, pandas_result)

    try:
        pandas_result = pandas_df.T.any(axis=axis, skipna=skipna, bool_only=bool_only)
    except Exception as e:
        with pytest.raises(type(e)):
            modin_df.T.any(axis=axis, skipna=skipna, bool_only=bool_only)
    else:
        modin_result = modin_df.T.any(axis=axis, skipna=skipna, bool_only=bool_only)
        df_equals(modin_result, pandas_result)

    try:
        pandas_result = pandas_df.T.any(axis=None, skipna=skipna, bool_only=bool_only)
    except Exception as e:
        with pytest.raises(type(e)):
            modin_df.T.any(axis=None, skipna=skipna, bool_only=bool_only)
    else:
        modin_result = modin_df.T.any(axis=None, skipna=skipna, bool_only=bool_only)
        df_equals(modin_result, pandas_result)

    # test level
    modin_df_multi_level = modin_df.copy()
    pandas_df_multi_level = pandas_df.copy()
    axis = modin_df._get_axis_number(axis) if axis is not None else 0
    levels = 3
    axis_names_list = [["a", "b", "c"], None]
    for axis_names in axis_names_list:
        if axis == 0:
            new_idx = pandas.MultiIndex.from_tuples(
                [(i // 4, i // 2, i) for i in range(len(modin_df.index))],
                names=axis_names,
            )
            modin_df_multi_level.index = new_idx
            pandas_df_multi_level.index = new_idx
        else:
            new_col = pandas.MultiIndex.from_tuples(
                [(i // 4, i // 2, i) for i in range(len(modin_df.columns))],
                names=axis_names,
            )
            modin_df_multi_level.columns = new_col
            pandas_df_multi_level.columns = new_col

        for level in list(range(levels)) + (axis_names if axis_names else []):
            try:
                pandas_multi_level_result = pandas_df_multi_level.any(
                    axis=axis, bool_only=bool_only, level=level, skipna=skipna
                )

            except Exception as e:
                with pytest.raises(type(e)):
                    modin_df_multi_level.any(
                        axis=axis, bool_only=bool_only, level=level, skipna=skipna
                    )
            else:
                modin_multi_level_result = modin_df_multi_level.any(
                    axis=axis, bool_only=bool_only, level=level, skipna=skipna
                )

                df_equals(modin_multi_level_result, pandas_multi_level_result)
