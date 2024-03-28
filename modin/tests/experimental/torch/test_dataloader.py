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

from types import ModuleType
from typing import Type

import numpy as np
import pandas
import pytest
import ray
import torch
from torch.utils.data import RandomSampler, Sampler, SequentialSampler

import modin.pandas as pd
from modin.experimental.torch.datasets import ModinDataLoader


@pytest.fixture(scope="module", autouse=True)
def ray_fix():
    ray.init(num_cpus=1)
    yield None
    ray.shutdown()


def _load_test_dataframe(lib: ModuleType):
    df = lib.read_csv(
        "https://raw.githubusercontent.com/ponder-org/ponder-datasets/main/USA_Housing.csv"
    )
    return df


@pytest.mark.parametrize("lib", [pandas, pd])
@pytest.mark.parametrize("sampler_cls", [RandomSampler, SequentialSampler])
@pytest.mark.parametrize("batch_size", [16, 37])
def test_torch_dataloader(lib: ModuleType, sampler_cls: Type[Sampler], batch_size: int):
    df = _load_test_dataframe(lib)
    np.random.seed(42)
    torch.manual_seed(42)
    loader = ModinDataLoader(
        df,
        batch_size=batch_size,
        features=[
            "AVG_AREA_INCOME",
            "AVG_AREA_HOUSE_AGE",
            "AVG_AREA_NUM_ROOMS",
            "AVG_AREA_NUM_BEDROOMS",
            "POPULATION",
            "PRICE",
        ],
        sampler=sampler_cls,
    )

    outputs = []
    for batch in loader:
        assert batch.shape[0] <= batch_size, batch.shape
        assert batch.shape[1] == 6, batch.shape

        outputs.append(batch)

    return outputs


@pytest.mark.parametrize("sampler_cls", [RandomSampler, SequentialSampler])
@pytest.mark.parametrize("batch_size", [16, 37])
def test_compare_dataloaders(sampler_cls: Type[Sampler], batch_size: int):
    by_modin = test_torch_dataloader(pd, sampler_cls, batch_size=batch_size)
    by_pandas = test_torch_dataloader(pandas, sampler_cls, batch_size=batch_size)

    assert len(by_modin) == len(by_pandas)
    for tensor_by_modin, tensor_by_pandas in zip(by_modin, by_pandas):
        assert np.allclose(tensor_by_modin, tensor_by_pandas), (
            tensor_by_modin - tensor_by_pandas
        )
