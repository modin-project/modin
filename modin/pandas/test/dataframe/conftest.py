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

from modin.config import StorageFormat


def pytest_collection_modifyitems(items):
    if StorageFormat.get() == "Hdk":
        for item in items:
            if item.name in (
                "test_sum[data0-over_rows_int-skipna_True-True]",
                "test_sum[data0-over_rows_str-skipna_True-True]",
            ):
                item.add_marker(
                    pytest.mark.xfail(
                        reason="https://github.com/intel-ai/hdk/issues/286"
                    )
                )
            elif item.name == "test_insert_dtypes[category-int_data]":
                item.add_marker(
                    pytest.mark.xfail(
                        reason="Categorical columns are converted to string due to #1698"
                    )
                )
            elif item.name == "test_insert_dtypes[int32-float_nan_data]":
                item.add_marker(
                    pytest.mark.xfail(
                        reason="HDK does not raise IntCastingNaNError on NaN to int cast"
                    )
                )
