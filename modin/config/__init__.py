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

"""Module houses config entities which can be used for Modin behavior tuning."""

from modin.config.envvars import (
    AsvDataSizeConfig,
    AsvImplementation,
    AsyncReadMode,
    BenchmarkMode,
    CIAWSAccessKeyID,
    CIAWSSecretAccessKey,
    CpuCount,
    DaskThreadsPerWorker,
    DocModule,
    DoUseCalcite,
    Engine,
    EnvironmentVariable,
    ExperimentalGroupbyImpl,
    ExperimentalNumPyAPI,
    GithubCI,
    GpuCount,
    HdkFragmentSize,
    HdkLaunchParameters,
    IsDebug,
    IsExperimental,
    IsRayCluster,
    LazyExecution,
    LogFileSize,
    LogMemoryInterval,
    LogMode,
    Memory,
    MinPartitionSize,
    ModinNumpy,
    NPartitions,
    PersistentPickle,
    ProgressBar,
    RangePartitioning,
    RangePartitioningGroupby,
    RayRedisAddress,
    RayRedisPassword,
    ReadSqlEngine,
    StorageFormat,
    TestDatasetSize,
    TestReadFromPostgres,
    TestReadFromSqlServer,
    TrackFileLeaks,
    use_range_partitioning_groupby,
)
from modin.config.pubsub import Parameter, ValueSource, context

__all__ = [
    "EnvironmentVariable",
    "Parameter",
    "ValueSource",
    "context",
    # General settings
    "IsDebug",
    "Engine",
    "StorageFormat",
    "CpuCount",
    "GpuCount",
    "Memory",
    # Ray specific
    "IsRayCluster",
    "RayRedisAddress",
    "RayRedisPassword",
    "LazyExecution",
    # Dask specific
    "DaskThreadsPerWorker",
    # Partitioning
    "NPartitions",
    "MinPartitionSize",
    # HDK specific
    "HdkFragmentSize",
    "DoUseCalcite",
    "HdkLaunchParameters",
    # ASV specific
    "TestDatasetSize",
    "AsvImplementation",
    "AsvDataSizeConfig",
    # Specific features
    "ProgressBar",
    "BenchmarkMode",
    "PersistentPickle",
    "ModinNumpy",
    "ExperimentalNumPyAPI",
    "RangePartitioningGroupby",
    "RangePartitioning",
    "use_range_partitioning_groupby",
    "ExperimentalGroupbyImpl",
    "AsyncReadMode",
    "ReadSqlEngine",
    "IsExperimental",
    # For tests
    "TrackFileLeaks",
    "TestReadFromSqlServer",
    "TestReadFromPostgres",
    "GithubCI",
    "CIAWSSecretAccessKey",
    "CIAWSAccessKeyID",
    # Logging
    "LogMode",
    "LogMemoryInterval",
    "LogFileSize",
    # Plugin settings
    "DocModule",
]
