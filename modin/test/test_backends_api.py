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

from modin.backends import BaseQueryCompiler, PandasQueryCompiler, PyarrowQueryCompiler

import pytest

BASE_BACKEND = BaseQueryCompiler
BACKENDS = [PandasQueryCompiler, PyarrowQueryCompiler]


def test_base_abstract_methods():
    allowed_abstract_methods = [
        "__init__",
        "free",
        "to_pandas",
        "from_pandas",
        "from_arrow",
        "default_to_pandas",
    ]

    not_implemented_methods = BASE_BACKEND.__abstractmethods__.difference(
        allowed_abstract_methods
    )

    # sorting for beauty output in error
    not_implemented_methods = list(not_implemented_methods)
    not_implemented_methods.sort()

    assert (
        len(not_implemented_methods) == 0
    ), f"{BASE_BACKEND} has not implemented abstract methods: {not_implemented_methods}"


@pytest.mark.parametrize("backend", BACKENDS)
def test_api_consistent(backend):
    base_methods = set(BASE_BACKEND.__dict__)
    custom_methods = set(
        [key for key in backend.__dict__.keys() if not key.startswith("_")]
    )

    extra_methods = custom_methods.difference(base_methods)
    # checking that custom backend do not implements extra api methods
    assert (
        len(extra_methods) == 0
    ), f"{backend} implement these extra methods: {extra_methods}"
