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

import pickle

import numpy as np
import pytest

import modin.pandas as pd
from modin.config import PersistentPickle
from modin.tests.pandas.utils import create_test_dfs, df_equals


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
    "modin_df", [pytest.param(modin_df), pytest.param(pd.DataFrame(), id="empty_df")]
)
def test_dataframe_pickle(modin_df, persistent):
    other = pickle.loads(pickle.dumps(modin_df))
    df_equals(modin_df, other)


def test__reduce__():
    # `DataFrame.__reduce__` will be called implicitly when lambda expressions are
    # pre-processed for the distributed engine.
    dataframe_data = ["Major League Baseball", "National Basketball Association"]
    abbr_md, abbr_pd = create_test_dfs(dataframe_data, index=["MLB", "NBA"])

    dataframe_data = {
        "name": ["Mariners", "Lakers"] * 500,
        "league_abbreviation": ["MLB", "NBA"] * 500,
    }
    teams_md, teams_pd = create_test_dfs(dataframe_data)

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


def test_column_pickle(modin_column, modin_df, persistent):
    dmp = pickle.dumps(modin_column)
    other = pickle.loads(dmp)
    df_equals(modin_column.to_frame(), other.to_frame())

    # make sure we don't pickle the whole frame if doing persistent storage
    if persistent:
        assert len(dmp) < len(pickle.dumps(modin_df))
