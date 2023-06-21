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
from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import torch
from torch import Tensor

from modin.experimental.torch.datasets import to_dataloader

if TYPE_CHECKING:
    from typing import ModuleType


def _load_test_dataframe(pandas: ModuleType):
    df = pandas.read_csv(
        "https://raw.githubusercontent.com/ponder-org/ponder-datasets/main/USA_Housing.csv"
    )
    features = df[
        [
            "AVG_AREA_INCOME",
            "AVG_AREA_HOUSE_AGE",
            "AVG_AREA_NUM_ROOMS",
            "AVG_AREA_NUM_BEDROOMS",
            "POPULATION",
            "PRICE",
        ]
    ]
    return features


def _test_torch_dataloader(pandas: ModuleType, with_index: bool):
    df = _load_test_dataframe(pandas)
    loader = to_dataloader(df, batch_size=16, with_index=with_index)
    for batch in loader:
        if with_index:
            idx, batch = batch
            assert isinstance(idx, Tensor)
            assert idx.dtype in {int, torch.int, torch.int32, torch.int64}
        assert batch.shape[0] <= 16, batch.shape
        assert batch.shape[1] == 6, batch.shape


def test_pandas_torch_dataloader():
    _test_torch_dataloader(pd, False)
    _test_torch_dataloader(pd, True)
