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

from modin.core.execution.client.query_compiler import ClientQueryCompiler
from modin.core.storage_formats import (
    BaseQueryCompiler,
    PandasQueryCompiler,
)
from modin.experimental.core.storage_formats.pyarrow import PyarrowQueryCompiler


BASE_EXECUTION = BaseQueryCompiler
EXECUTIONS = [PandasQueryCompiler, PyarrowQueryCompiler, ClientQueryCompiler]


def test_base_abstract_methods():
    allowed_abstract_methods = [
        "__init__",
        "free",
        "finalize",
        "to_pandas",
        "from_pandas",
        "from_arrow",
        "default_to_pandas",
        "from_dataframe",
        "to_dataframe",
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


@pytest.mark.parametrize(
    "execution,expected_extra_methods",
    [
        (PandasQueryCompiler, set()),
        (PyarrowQueryCompiler, set()),
        # client query compiler exposes set_server_connection,
        # which the other compilers should not
        (ClientQueryCompiler, {"set_server_connection"}),
    ],
)
def test_api_consistent(execution, expected_extra_methods):
    base_methods = set(BASE_EXECUTION.__dict__)
    custom_methods = set(
        [key for key in execution.__dict__.keys() if not key.startswith("_")]
    )

    extra_methods = custom_methods.difference(base_methods)
    assert (
        extra_methods == expected_extra_methods
    ), f"{execution} implement these extra methods: {extra_methods}"
