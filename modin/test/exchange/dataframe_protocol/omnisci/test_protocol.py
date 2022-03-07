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

"""Dataframe exchange protocol tests that are specific for OmniSci implementation."""

from modin.experimental.core.execution.native.implementations.omnisci_on_native.exchange.dataframe_protocol.__utils import (
    from_dataframe,
)
from modin.experimental.core.execution.native.implementations.omnisci_on_native.test.utils import (
    ForceOmnisciImport,
)

import pytest
import modin.pandas as pd
import pandas
import numpy as np

from modin.pandas.test.utils import df_equals


def get_all_types(has_nulls=False):
    int_data = {}
    uint_data = {}
    float_data = {}
    datetime_data = {}
    string_data = {}
    category_data = {}

    # int
    for width in (8, 16, 32, 64):
        dtype = getattr(np, f"int{width}")
        max_val, min_val = np.iinfo(dtype).max, np.iinfo(dtype).min
        int_data[f"int{width}_col"] = np.array(
            [max_val, max_val - 1, min_val, min_val + 1] * 10, dtype=dtype
        )

    # uint
    for width in (8, 16, 32, 64):
        dtype = getattr(np, f"uint{width}")
        max_val, min_val = np.iinfo(dtype).max, np.iinfo(dtype).min
        uint_data[f"uint{width}_col"] = np.array(
            [max_val, max_val - 1, min_val, min_val + 1] * 10, dtype=dtype
        )

    # float
    for width in (32, 64):
        dtype = getattr(np, f"float{width}")
        max_val, min_val = np.finfo(dtype).max, np.finfo(dtype).min
        float_data[f"float{width}_col"] = np.array(
            [max_val, max_val - 1, min_val, min_val + 1] * 10, dtype=dtype
        )
        if has_nulls:
            float_data[f"float{width}_null_col"] = np.array(
                [max_val, None, min_val, min_val + 1] * 10, dtype=dtype
            )

    # datetime
    for unit in ("s", "ms", "ns"):
        datetime_data[f"datetime64[{unit}]_col"] = np.array(
            [0, 1, 2, 3] * 10, dtype=np.dtype(f"datetime64[{unit}]")
        )
        if has_nulls:
            datetime_data[f"datetime64[{unit}]_null_col"] = np.array(
                [0, None, 2, 3] * 10, dtype=np.dtype(f"datetime64[{unit}]")
            )

    # string
    string_data["string_col"] = np.array(["Sample", "te", "", "xt"] * 10)
    if has_nulls:
        string_data["string_null_col"] = np.array(["Sample", None, "", "xt"] * 10)

    # category
    category_data["category_int_col"] = pandas.Categorical([1, 2, 3, 4] * 10)
    category_data["category_string_col"] = pandas.Categorical(
        ["Sample", "te", "", "xt"] * 10
    )
    if has_nulls:
        category_data["category_string_null_col"] = pandas.Categorical(
            ["Sample", None, "", "xt"] * 10
        )

    return {
        **int_data,
        **uint_data,
        **float_data,
        **datetime_data,
        **string_data,
        **category_data,
    }


@pytest.mark.parametrize("data_has_nulls", [True, False])
def test_simple_export(data_has_nulls):
    data = get_all_types(has_nulls=data_has_nulls)
    md_df = pd.DataFrame(data)
    exported_df = from_dataframe(md_df._query_compiler._modin_frame)
    df_equals(md_df, exported_df)

    exported_df = from_dataframe(md_df._query_compiler._modin_frame, nchunks=3)
    df_equals(md_df, exported_df)

    exported_df = from_dataframe(md_df._query_compiler._modin_frame, nchunks=5)
    df_equals(md_df, exported_df)

    exported_df = from_dataframe(md_df._query_compiler._modin_frame, nchunks=12)
    df_equals(md_df, exported_df)


# @pytest.mark.parametrize("data_has_nulls", [True, False])
# def test_export_from_omnisci(data_has_nulls):
#     data = get_all_types(has_nulls=data_has_nulls)
#     md_df = pd.DataFrame(data)

#     with ForceOmnisciImport(md_df) as instance:
#         # md_df_exported
#         exported_df = from_dataframe(md_df._query_compiler._modin_frame)
#     df_equals(md_df, exported_df)

#     exported_df = from_dataframe(md_df._query_compiler._modin_frame, nchunks=3)
#     df_equals(md_df, exported_df)

#     exported_df = from_dataframe(md_df._query_compiler._modin_frame, nchunks=5)
#     df_equals(md_df, exported_df)

#     exported_df = from_dataframe(md_df._query_compiler._modin_frame, nchunks=12)
#     df_equals(md_df, exported_df)
