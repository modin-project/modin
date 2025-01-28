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

from modin.core.storage_formats import BaseQueryCompiler, PandasQueryCompiler

BASE_EXECUTION = BaseQueryCompiler
EXECUTIONS = [PandasQueryCompiler]


def test_base_abstract_methods():
    allowed_abstract_methods = [
        "__init__",
        "free",
        "finalize",
        "execute",
        "to_pandas",
        "from_pandas",
        "from_arrow",
        "default_to_pandas",
        "from_interchange_dataframe",
        "to_interchange_dataframe",
        "engine",
        "storage_format",
    ]

    not_implemented_methods = BASE_EXECUTION.__abstractmethods__.difference(
        allowed_abstract_methods
    )

    # sorting for beauty output in error
    not_implemented_methods = list(not_implemented_methods)
    not_implemented_methods.sort()

    assert (
        len(not_implemented_methods) == 0
    ), f"{BASE_EXECUTION} has not implemented abstract methods: {not_implemented_methods}"


@pytest.mark.parametrize("execution", EXECUTIONS)
def test_api_consistent(execution):
    base_methods = set(BASE_EXECUTION.__dict__)
    custom_methods = set(
        [key for key in execution.__dict__.keys() if not key.startswith("_")]
    )

    extra_methods = custom_methods.difference(base_methods)
    # checking that custom execution do not implements extra api methods
    assert (
        len(extra_methods) == 0
    ), f"{execution} implement these extra methods: {extra_methods}"
