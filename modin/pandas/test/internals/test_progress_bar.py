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

import unittest.mock as mock

import numpy as np
import pandas
import pytest

from modin.config import NPartitions
from modin.pandas.utils import from_pandas
from modin.utils import get_current_execution

NParts = 4
NPartitions.put(NParts)


@pytest.mark.skipif(get_current_execution() != "PandasOnRay", reason="not supported")
def test_progress_bar_does_not_broke_execution1(enable_benchmark_mode):
    data = pandas.DataFrame(np.random.rand(2**10, 2**6))
    tqdm_instance = mock.MagicMock()
    with mock.patch("tqdm.autonotebook.tqdm", return_value=tqdm_instance) as tqdm_class:
        _ = from_pandas(data)
        tqdm_class.assert_called_once()
        # have to multiply by 2 to take into account column partitions
        assert tqdm_class.call_args.kwargs["total"] == NParts * 2
        # + 1 for `close` method
        assert len(tqdm_instance.mock_calls) == tqdm_class.call_args.kwargs["total"] + 1
        for call in tqdm_instance.method_calls[:-1]:
            assert str(call).endswith("update(1)")
        assert str(tqdm_instance.method_calls[-1]).endswith("close()")


@pytest.mark.skipif(get_current_execution() != "PandasOnRay", reason="not supported")
def test_progress_bar_does_not_broke_execution2(enable_benchmark_mode):
    from modin.core.execution.modin_aqp import call_progress_bar

    data = pandas.DataFrame(np.random.rand(2**10, 2**6))
    modin_df = from_pandas(data)
    internal_frame = modin_df._query_compiler._modin_frame
    with mock.patch("threading.Thread") as thread_class:
        return_value = internal_frame._partition_mgr_cls.map_partitions(
            internal_frame._partitions, lambda part: part
        )
        assert thread_class.mock_calls[0].kwargs["target"] == call_progress_bar
        assert (thread_class.mock_calls[0].kwargs["args"][0] == return_value).all()
