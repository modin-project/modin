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

import warnings

import matplotlib
import pytest

import modin.pandas as pd
from modin.config import NPartitions
from modin.pandas.utils import SET_DATAFRAME_ATTRIBUTE_WARNING
from modin.tests.pandas.native_df_interoperability.utils import (
    create_test_df_in_defined_mode,
    create_test_series_in_defined_mode,
)
from modin.tests.pandas.utils import df_equals, eval_general

NPartitions.put(4)

# Force matplotlib to not use any Xwindows backend.
matplotlib.use("Agg")


def test___setattr__mutating_column(df_mode_pair):
    # Use case from issue #4577
    modin_df, pandas_df = create_test_df_in_defined_mode(
        [[1]], columns=["col0"], native=df_mode_pair[0]
    )
    # Replacing a column with a list should mutate the column in place.
    pandas_df.col0 = [3]
    modin_df.col0 = [3]
    modin_ser, pandas_ser = create_test_series_in_defined_mode(
        [3], native=df_mode_pair[1]
    )
    df_equals(modin_df, pandas_df)
    # Check that the col0 attribute reflects the value update.
    df_equals(modin_df.col0, pandas_df.col0)

    pandas_df.col0 = pandas_ser
    modin_df.col0 = modin_ser

    # Check that the col0 attribute reflects this update
    df_equals(modin_df, pandas_df)

    pandas_df.loc[0, "col0"] = 4
    modin_df.loc[0, "col0"] = 4

    # Check that the col0 attribute reflects update via loc
    df_equals(modin_df, pandas_df)
    assert modin_df.col0.equals(modin_df["col0"])

    # Check that attempting to add a new col via attributes raises warning
    # and adds the provided list as a new attribute and not a column.
    with pytest.warns(
        UserWarning,
        match=SET_DATAFRAME_ATTRIBUTE_WARNING,
    ):
        modin_df.col1 = [4]

    with warnings.catch_warnings():
        warnings.filterwarnings(
            action="error",
            message=SET_DATAFRAME_ATTRIBUTE_WARNING,
        )
        modin_df.col1 = [5]
        modin_df.new_attr = 6
        modin_df.col0 = 7

    assert "new_attr" in dir(
        modin_df
    ), "Modin attribute was not correctly added to the df."
    assert (
        "new_attr" not in modin_df
    ), "New attribute was not correctly added to columns."
    assert modin_df.new_attr == 6, "Modin attribute value was set incorrectly."
    assert isinstance(
        modin_df.col0, pd.Series
    ), "Scalar was not broadcasted properly to an existing column."


def test_isin_with_modin_objects(df_mode_pair):
    modin_df1, pandas_df1 = create_test_df_in_defined_mode(
        {"a": [1, 2], "b": [3, 4]}, native=df_mode_pair[0]
    )
    modin_series, pandas_series = create_test_series_in_defined_mode(
        [1, 4, 5, 6], native=df_mode_pair[1]
    )

    eval_general(
        (modin_df1, modin_series),
        (pandas_df1, pandas_series),
        lambda srs: srs[0].isin(srs[1]),
    )

    modin_df2 = modin_series.to_frame("a")
    pandas_df2 = pandas_series.to_frame("a")

    eval_general(
        (modin_df1, modin_df2),
        (pandas_df1, pandas_df2),
        lambda srs: srs[0].isin(srs[1]),
    )

    # Check case when indices are not matching
    modin_df1, pandas_df1 = create_test_df_in_defined_mode(
        {"a": [1, 2], "b": [3, 4]},
        index=[10, 11],
        native=df_mode_pair[0],
    )

    eval_general(
        (modin_df1, modin_series),
        (pandas_df1, pandas_series),
        lambda srs: srs[0].isin(srs[1]),
    )
    eval_general(
        (modin_df1, modin_df2),
        (pandas_df1, pandas_df2),
        lambda srs: srs[0].isin(srs[1]),
    )
