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

"""Compatibility layer for parameters used by ASV."""

import os

import modin.pandas as pd

try:
    from modin.config import NPartitions

    NPARTITIONS = NPartitions.get()
except ImportError:
    NPARTITIONS = pd.DEFAULT_NPARTITIONS

try:
    from modin.config import AsvImplementation, Engine, StorageFormat, TestDatasetSize

    ASV_USE_IMPL = AsvImplementation.get()
    ASV_DATASET_SIZE = TestDatasetSize.get() or "Small"
    ASV_USE_ENGINE = Engine.get()
    ASV_USE_STORAGE_FORMAT = StorageFormat.get()
except ImportError:
    # The same benchmarking code can be run for different versions of Modin, so in
    # case of an error importing important variables, we'll just use predefined values
    ASV_USE_IMPL = os.environ.get("MODIN_ASV_USE_IMPL", "modin")
    ASV_DATASET_SIZE = os.environ.get("MODIN_TEST_DATASET_SIZE", "Small")
    ASV_USE_ENGINE = os.environ.get("MODIN_ENGINE", "Ray")
    ASV_USE_STORAGE_FORMAT = os.environ.get("MODIN_STORAGE_FORMAT", "Pandas")

ASV_USE_IMPL = ASV_USE_IMPL.lower()
ASV_DATASET_SIZE = ASV_DATASET_SIZE.lower()
ASV_USE_ENGINE = ASV_USE_ENGINE.lower()
ASV_USE_STORAGE_FORMAT = ASV_USE_STORAGE_FORMAT.lower()

assert ASV_USE_IMPL in ("modin", "pandas")
assert ASV_DATASET_SIZE in ("big", "small")
assert ASV_USE_ENGINE in ("ray", "dask", "python", "unidist")
assert ASV_USE_STORAGE_FORMAT in ("pandas")
