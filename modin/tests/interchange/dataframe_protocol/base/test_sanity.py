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

"""Basic sanity checks for the DataFrame exchange protocol."""

import pytest

import modin.pandas as pd
from modin.tests.pandas.utils import default_to_pandas_ignore_string


def test_sanity():
    """Test that the DataFrame protocol module is valid and could be imported correctly."""
    from modin.core.dataframe.base.interchange.dataframe_protocol.dataframe import (  # noqa
        ProtocolDataframe,
    )


@pytest.mark.filterwarnings(default_to_pandas_ignore_string)
def test_basic_io(get_unique_base_execution):
    """Test that the protocol IO functions actually reach their implementation with no errors."""

    class TestPassed(BaseException):
        pass

    def dummy_io_method(*args, **kwargs):
        """Dummy method emulating that the code path reached the exchange protocol implementation."""
        raise TestPassed

    query_compiler_cls = get_unique_base_execution
    query_compiler_cls.from_dataframe = dummy_io_method
    query_compiler_cls.to_dataframe = dummy_io_method

    from modin.pandas.io import from_dataframe

    with pytest.raises(TestPassed):
        from_dataframe(None)

    with pytest.raises(TestPassed):
        pd.DataFrame([[1]]).__dataframe__()
