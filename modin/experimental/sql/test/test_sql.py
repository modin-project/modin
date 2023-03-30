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
import modin.pandas as pd
import modin.config as cfg
from modin.pandas.test.utils import default_to_pandas_ignore_string, df_equals

import io
import pytest

pytestmark = pytest.mark.filterwarnings(default_to_pandas_ignore_string)

titanic_snippet = """passenger_id,survived,p_class,name,sex,age,sib_sp,parch,ticket,fare,cabin,embarked
1,0,3,"Braund, Mr. Owen Harris",male,22,1,0,A/5 21171,7.25,,S
2,1,1,"Cumings, Mrs. John Bradley (Florence Briggs Thayer)",female,38,1,0,PC 17599,71.2833,C85,C
3,1,3,"Heikkinen, Miss. Laina",female,26,0,0,STON/O2. 3101282,7.925,,S
4,1,1,"Futrelle, Mrs. Jacques Heath (Lily May Peel)",female,35,1,0,113803,53.1,C123,S
5,0,3,"Allen, Mr. William Henry",male,35,0,0,373450,8.05,,S
6,0,3,"Moran, Mr. James",male,,0,0,330877,8.4583,,Q
7,0,1,"McCarthy, Mr. Timothy J",male,54,0,0,17463,51.8625,E46,S
8,0,3,"Palsson, Master. Gosta Leonard",male,2,3,1,349909,21.075,,S
9,1,3,"Johnson, Mrs. Oscar W (Elisabeth Vilhelmina Berg)",female,27,0,2,347742,11.1333,,S
"""


def test_sql_query():
    from modin.experimental.sql import query

    df = pd.read_csv(io.StringIO(titanic_snippet))
    sql = "SELECT survived, p_class, count(passenger_id) as cnt FROM (SELECT * FROM titanic WHERE survived = 1) as t1 GROUP BY survived, p_class"
    query_result = query(sql, titanic=df)
    expected_df = (
        df[df.survived == 1]
        .groupby(["survived", "p_class"])
        .agg({"passenger_id": "count"})
        .reset_index()
    )
    assert query_result.shape == expected_df.shape
    values_left = expected_df.dropna().values
    values_right = query_result.dropna().values
    assert (values_left == values_right).all()


def test_sql_extension():
    # This test is for DataFrame.sql() method, that is injected by
    # dfsql.extensions. In the HDK environment, there is no dfsql
    # module and, thus, this test fails.
    if cfg.StorageFormat.get() == "Hdk":
        return

    import modin.experimental.sql  # noqa: F401

    df = pd.read_csv(io.StringIO(titanic_snippet))

    expected_df = df[df["survived"] == 1][["passenger_id", "survived"]]

    sql = "SELECT passenger_id, survived WHERE survived = 1"
    query_result = df.sql(sql)
    assert list(query_result.columns) == ["passenger_id", "survived"]
    values_left = expected_df.values
    values_right = query_result.values
    assert values_left.shape == values_right.shape
    assert (values_left == values_right).all()


def test_string_cast():
    from modin.experimental.sql import query

    data = {"A": ["A", "B", "C"], "B": ["A", "B", "C"]}
    mdf = pd.DataFrame(data)
    pdf = pandas.DataFrame(data)
    df_equals(pdf, query("SELECT * FROM df", df=mdf))
