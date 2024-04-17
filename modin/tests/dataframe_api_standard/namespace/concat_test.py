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


# MIT License

# Copyright (c) 2023, Marco Gorelli

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


from __future__ import annotations

import pytest

from modin.dataframe_api_standard.tests.utils import (
    BaseHandler,
    compare_dataframe_with_reference,
    integer_dataframe_1,
    integer_dataframe_2,
    integer_dataframe_4,
)


def test_concat(library: BaseHandler) -> None:
    df1 = integer_dataframe_1(library)
    df2 = integer_dataframe_2(library)
    ns = df1.__dataframe_namespace__()
    result = ns.concat([df1, df2])
    expected = {"a": [1, 2, 3, 1, 2, 4], "b": [4, 5, 6, 4, 2, 6]}
    compare_dataframe_with_reference(result, expected, dtype=ns.Int64)


def test_concat_mismatch(library: BaseHandler) -> None:
    df1 = integer_dataframe_1(library).persist()
    df2 = integer_dataframe_4(library).persist()
    ns = df1.__dataframe_namespace__()
    # TODO check the error
    with pytest.raises(ValueError):
        _ = ns.concat([df1, df2]).persist()
