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

import modin.pandas


def test_dataframe_api_standard() -> None:
    """
    Test some basic methods of the dataframe consortium standard.

    Full testing is done at https://github.com/data-apis/dataframe-api-compat,
    this is just to check that the entry point works as expected.
    """
    pytest.importorskip("dataframe_api_compat")
    df_pd = modin.pandas.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    df = df_pd.__dataframe_consortium_standard__()
    result_1 = df.get_column_names()
    expected_1 = ["a", "b"]
    assert result_1 == expected_1

    ser = modin.pandas.Series([1, 2, 3])
    col = ser.__column_consortium_standard__()
    result_2 = col.get_value(1)
    expected_2 = 2
    assert result_2 == expected_2
