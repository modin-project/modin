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

import os

import pytest


@pytest.fixture
def nameset():
    name = "hey_i_am_an_env_var"
    os.environ[name] = "i am a value"
    yield name
    del os.environ[name]


def test_envvar_catcher(nameset):
    with pytest.raises(AssertionError):
        os.environ.get("Modin_FOO", "bar")
    with pytest.raises(AssertionError):
        "modin_qux" not in os.environ
    assert "yay_random_name" not in os.environ
    assert os.environ[nameset]
