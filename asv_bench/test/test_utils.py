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

from unittest.mock import Mock, mock_open, patch

import numpy as np
import pytest
from benchmarks.utils import data_shapes, execute, get_benchmark_shapes

import modin.pandas as pd
from modin.config import AsvDataSizeConfig


@pytest.mark.parametrize(
    "asv_config_content, result",
    [
        (
            '{"TimeJoin": [[[10, 10], [15, 15]], [[11, 11], [13, 13]]], \
                "TimeGroupBy": [[11, 11], [13, 13]]}',
            [
                [
                    # binary shapes
                    [[10, 10], [15, 15]],
                    [[11, 11], [13, 13]],
                ],
                [
                    # unary shapes
                    [11, 11],
                    [13, 13],
                ],
            ],
        ),
    ],
)
@patch.object(data_shapes, "CONFIG_FROM_FILE", new=None)
def test_get_benchmark_shapes(asv_config_content, result):
    AsvDataSizeConfig.put("mock_filename")
    with patch("builtins.open", mock_open(read_data=asv_config_content)):
        assert result[0] == get_benchmark_shapes("TimeJoin")
        assert result[1] == get_benchmark_shapes("TimeGroupBy")


@pytest.mark.parametrize(
    "asv_config_content, result",
    [
        (
            '{"TimeJoin": [[[10, 10], [15, 15]]]',
            [[100, 100]],
        ),
    ],
)
@patch.object(data_shapes, "CONFIG_FROM_FILE", new=None)
def test_get_benchmark_shapes_default(asv_config_content, result):
    AsvDataSizeConfig.put(None)
    with patch.object(data_shapes, "DEFAULT_CONFIG", new={"TimeJoin": result}):
        assert result == get_benchmark_shapes("TimeJoin")


def test_execute():
    df = pd.DataFrame(np.random.rand(100, 64))
    partitions = df._query_compiler._modin_frame._partitions.flatten()
    mgr_cls = df._query_compiler._modin_frame._partition_mgr_cls
    with patch.object(mgr_cls, "wait_partitions", new=Mock()):
        execute(df)
        mgr_cls.wait_partitions.assert_called_once()
        assert (mgr_cls.wait_partitions.call_args[0] == partitions).all()
