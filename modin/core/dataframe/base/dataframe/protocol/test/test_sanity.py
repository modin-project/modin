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

"""Perform basic sanity checks for the DataFrame exchange protocol."""

import pytest
import modin.pandas as pd


def test_sanity():
    """Test that the DataFrame protocol module is valid and could be imported correctly."""
    from modin.core.dataframe.base.dataframe.protocol.dataframe import (  # noqa
        ProtocolDataframe,
    )


def test_basic_io():
    """Test that the protocol IO functions actually reach their implementation with no errors."""

    class TestPassed(BaseException):
        pass

    def dummy_io_method(*args, **kwargs):
        """Dummy method emulating that the code path reached exchange protocol implementation."""
        raise TestPassed

    from modin.core.storage_formats.base.query_compiler import BaseQueryCompiler

    BaseQueryCompiler.from_dataframe = dummy_io_method
    BaseQueryCompiler.to_dataframe = dummy_io_method

    from modin.pandas.utils import from_dataframe

    with pytest.raises(TestPassed):
        from_dataframe(None)

    with pytest.raises(TestPassed):
        pd.DataFrame([[1]]).__dataframe__()
