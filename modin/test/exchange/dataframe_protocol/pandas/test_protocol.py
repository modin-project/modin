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

import modin.pandas as pd
from modin.pandas.utils import from_dataframe
from modin.pandas.test.utils import df_equals, test_data
from modin.test.test_utils import warns_that_defaulting_to_pandas


def test_simple_import():
    modin_df_producer = pd.DataFrame(test_data["int_data"])
    internal_modin_df_producer = modin_df_producer.__dataframe__()
    # Our configuration in pytest.ini requires that we explicitly catch all
    # instances of defaulting to pandas, this one raises a warning on `.from_dataframe`
    with warns_that_defaulting_to_pandas():
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
