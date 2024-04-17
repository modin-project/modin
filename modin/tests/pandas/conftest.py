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

from modin.config import Engine, StorageFormat


def pytest_collection_modifyitems(items):
    try:
        if (
            Engine.get() in ("Ray", "Unidist", "Dask", "Python")
            and StorageFormat.get() != "Base"
        ):
            for item in items:
                if item.name in (
                    "test_dataframe_dt_index[3s-both-DateCol-_NoDefault.no_default]",
                    "test_dataframe_dt_index[3s-right-DateCol-_NoDefault.no_default]",
                ):
                    item.add_marker(
                        pytest.mark.xfail(
                            reason="https://github.com/modin-project/modin/issues/6399"
                        )
                    )
    except ImportError:
        # No engine
        ...
