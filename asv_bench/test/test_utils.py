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
from unittest.mock import patch, mock_open

from benchmarks.utils import data_shapes
from benchmarks.utils import get_benchmark_shapes

from modin.config import AsvDataSizeConfig


@pytest.mark.parametrize(
    "asv_config_content, result",
    [
        (
            '{"TimeJoin": ["((10, 10), (15,15))", "((11, 11), (13, 13))"], \
                "TimeMerge": ["((11, 11), (13, 13))"]}',
            [[((10, 10), (15, 15)), ((11, 11), (13, 13))], [((11, 11), (13, 13))]],
        ),
    ],
)
@patch.object(data_shapes, "CONFIG_FROM_FILE", new=None)
def test_get_benchmark_shapes(monkeypatch, asv_config_content, result):
    AsvDataSizeConfig.put("mock_filename")
    with patch("builtins.open", mock_open(read_data=asv_config_content)):
        assert result[0] == get_benchmark_shapes("TimeJoin")
        assert result[1] == get_benchmark_shapes("TimeMerge")


@pytest.mark.parametrize(
    "asv_config_content, result",
    [
        (
            '{"TimeJoin": ["((10, 10), (15,15))"]',
            [(100, 100)],
        ),
    ],
)
@patch.object(data_shapes, "CONFIG_FROM_FILE", new=None)
def test_get_benchmark_shapes_default(asv_config_content, result):
    AsvDataSizeConfig.put(None)
    with patch.object(data_shapes, "DEFAULT_CONFIG", new={"TimeJoin": result}):
        assert result == get_benchmark_shapes("TimeJoin")
