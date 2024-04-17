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

from modin.dataframe_api_standard.tests.utils import BaseHandler, mixed_dataframe_1


def test_schema(library: BaseHandler) -> None:
    df = mixed_dataframe_1(library)
    namespace = df.__dataframe_namespace__()
    result = df.schema
    assert list(result.keys()) == [
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        "g",
        "h",
        "i",
        "j",
        "k",
        "l",
        "m",
        "n",
        "o",
        "p",
        "q",
    ]
    assert isinstance(result["a"], namespace.Int64)
    assert isinstance(result["b"], namespace.Int32)
    assert isinstance(result["c"], namespace.Int16)
    assert isinstance(result["d"], namespace.Int8)
    assert isinstance(result["e"], namespace.UInt64)
    assert isinstance(result["f"], namespace.UInt32)
    assert isinstance(result["g"], namespace.UInt16)
    assert isinstance(result["h"], namespace.UInt8)
    assert isinstance(result["i"], namespace.Float64)
    assert isinstance(result["j"], namespace.Float32)
    assert isinstance(result["k"], namespace.Bool)
    assert isinstance(result["l"], namespace.String)
    assert isinstance(result["m"], namespace.Datetime)
    assert isinstance(result["n"], namespace.Datetime)
    assert result["n"].time_unit == "ms"
    assert result["n"].time_zone is None
    assert isinstance(result["o"], namespace.Datetime)
    assert result["o"].time_unit == "us"
    assert result["o"].time_zone is None
    # pandas non-nanosecond support only came in 2.0 - before that, these would be 'float'
    assert isinstance(result["p"], namespace.Duration)
    assert result["p"].time_unit == "ms"
    assert isinstance(result["q"], namespace.Duration)
    assert result["q"].time_unit == "us"
