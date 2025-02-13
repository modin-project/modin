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

"""Dataframe exchange protocol tests that are specific for pandas storage format implementation."""

import pandas

import modin.pandas as pd
from modin.pandas.io import from_dataframe
from modin.tests.pandas.utils import df_equals, test_data
from modin.tests.test_utils import (
    df_or_series_using_native_execution,
    warns_that_defaulting_to_pandas,
    warns_that_defaulting_to_pandas_if,
)


def eval_df_protocol(modin_df_producer):
    internal_modin_df_producer = modin_df_producer.__dataframe__()
    # Our configuration in pytest.ini requires that we explicitly catch all
    # instances of defaulting to pandas, this one raises a warning on `.from_dataframe`
    with warns_that_defaulting_to_pandas_if(
        not df_or_series_using_native_execution(modin_df_producer)
    ):
        modin_df_consumer = from_dataframe(modin_df_producer)
        internal_modin_df_consumer = from_dataframe(internal_modin_df_producer)

    # TODO: the following assertions verify that `from_dataframe` doesn't return
    # the same object untouched due to optimization branching, it actually should
    # do so but the logic is not implemented yet, so the assertions are passing
    # for now. It's required to replace the producer's type with a different one
    # to consumer when we have some other implementation of the protocol as the
    # assertions may start failing shortly.
    assert modin_df_producer is not modin_df_consumer
    assert internal_modin_df_producer is not internal_modin_df_consumer
    assert (
        modin_df_producer._query_compiler._modin_frame
        is not modin_df_consumer._query_compiler._modin_frame
    )

    df_equals(modin_df_producer, modin_df_consumer)
    df_equals(modin_df_producer, internal_modin_df_consumer)


def test_simple_import():
    modin_df = pd.DataFrame(test_data["int_data"])
    eval_df_protocol(modin_df)


def test_categorical_from_dataframe():
    modin_df = pd.DataFrame(
        {"foo": pd.Series(["0", "1", "2", "3", "0", "3", "2", "3"], dtype="category")}
    )
    eval_df_protocol(modin_df)


def test_from_dataframe_with_empty_dataframe():
    modin_df = pd.DataFrame({"foo_col": pd.Series([], dtype="int64")})
    with warns_that_defaulting_to_pandas():
        eval_df_protocol(modin_df)


def test_interchange_with_pandas_string():
    modin_df = pd.DataFrame({"fips": ["01001"]})
    pandas_df = pandas.api.interchange.from_dataframe(modin_df.__dataframe__())
    df_equals(modin_df, pandas_df)


def test_interchange_with_datetime():
    date_range = pd.date_range(
        start=pd.Timestamp("2024-01-01", unit="ns"),
        end=pd.Timestamp("2024-03-01", unit="ns"),
        freq="D",
    )
    modin_df = pd.DataFrame(
        {
            "datetime_s": date_range.astype("datetime64[s]"),
            "datetime_ns": date_range.astype("datetime64[ns]"),
        }
    )
    eval_df_protocol(modin_df)
