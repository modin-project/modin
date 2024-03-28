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

import numpy as np
import pandas
import pytest
from modin_spreadsheet import SpreadsheetWidget

import modin.experimental.spreadsheet as mss
import modin.pandas as pd


def get_test_data():
    return {
        "A": 1.0,
        "B": pd.Timestamp("20130102"),
        "C": pd.Series(1, index=list(range(4)), dtype="float32"),
        "D": np.array([5, 2, 3, 1], dtype="int32"),
        "E": pd.Categorical(["test", "train", "foo", "bar"]),
        "F": ["foo", "bar", "buzz", "fox"],
    }


def test_from_dataframe():
    data = get_test_data()
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    modin_result = mss.from_dataframe(modin_df)
    assert isinstance(modin_result, SpreadsheetWidget)

    with pytest.raises(TypeError):
        mss.from_dataframe(pandas_df)

    # Check parameters don't error
    def can_edit_row(row):
        return row["D"] > 2

    modin_result = mss.from_dataframe(
        modin_df,
        show_toolbar=True,
        show_history=True,
        precision=1,
        grid_options={"forceFitColumns": False, "filterable": False},
        column_options={"D": {"editable": True}},
        column_definitions={"editable": False},
        row_edit_callback=can_edit_row,
    )
    assert isinstance(modin_result, SpreadsheetWidget)


def test_to_dataframe():
    data = get_test_data()
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    spreadsheet = mss.from_dataframe(modin_df)
    modin_result = mss.to_dataframe(spreadsheet)

    assert modin_result.equals(modin_df)

    with pytest.raises(TypeError):
        mss.to_dataframe("Not a SpreadsheetWidget")
    with pytest.raises(TypeError):
        mss.to_dataframe(pandas_df)
