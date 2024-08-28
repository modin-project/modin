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

import numpy as np
import pytest

import modin.pandas as pd
from modin.config import NativeDataframeMode, PersistentPickle
from modin.tests.pandas.native_df_mode.utils import create_test_df_in_defined_mode
from modin.tests.pandas.utils import df_equals


@pytest.fixture
def modin_df():
    return pd.DataFrame({"col1": np.arange(1000), "col2": np.arange(2000, 3000)})


@pytest.fixture
def modin_column(modin_df):
    return modin_df["col1"]


@pytest.fixture(params=[True, False])
def persistent(request):
    old = PersistentPickle.get()
    PersistentPickle.put(request.param)
    yield request.param
    PersistentPickle.put(old)


@pytest.mark.parametrize(
    "df_mode_pair", list(product(NativeDataframeMode.choices, repeat=2))
)
def test__reduce__(df_mode_pair):
    # `DataFrame.__reduce__` will be called implicitly when lambda expressions are
    # pre-processed for the distributed engine.
    dataframe_data = ["Major League Baseball", "National Basketball Association"]
    abbr_md, abbr_pd = create_test_df_in_defined_mode(
        dataframe_data, index=["MLB", "NBA"], df_mode=df_mode_pair[0]
    )

    dataframe_data = {
        "name": ["Mariners", "Lakers"] * 500,
        "league_abbreviation": ["MLB", "NBA"] * 500,
    }
    teams_md, teams_pd = create_test_df_in_defined_mode(
        dataframe_data, df_mode=df_mode_pair[1]
    )

    result_md = (
        teams_md.set_index("name")
        .league_abbreviation.apply(lambda abbr: abbr_md[0].loc[abbr])
        .rename("league")
    )

    result_pd = (
        teams_pd.set_index("name")
        .league_abbreviation.apply(lambda abbr: abbr_pd[0].loc[abbr])
        .rename("league")
    )
    df_equals(result_md, result_pd)
