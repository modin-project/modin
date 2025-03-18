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

import pandas
import pytest

import modin.pandas as pd
from modin.core.storage_formats.base.query_compiler import QCCoercionCost
from modin.core.storage_formats.pandas.native_query_compiler import NativeQueryCompiler


class CloudQC(NativeQueryCompiler):
    "Represents a cloud-hosted query compiler"

    def __init__(self, pandas_frame):
        self._modin_frame = pandas_frame
        super().__init__(pandas_frame)

    def qc_engine_switch_cost(self, other_qc_cls):
        return {
            CloudQC: QCCoercionCost.COST_ZERO,
            ClusterQC: QCCoercionCost.COST_MEDIUM,
            DefaultQC: QCCoercionCost.COST_MEDIUM,
            LocalMachineQC: QCCoercionCost.COST_HIGH,
            PicoQC: QCCoercionCost.COST_IMPOSSIBLE,
        }[other_qc_cls]


class ClusterQC(NativeQueryCompiler):
    "Represents a local network cluster query compiler"

    def __init__(self, pandas_frame):
        self._modin_frame = pandas_frame
        super().__init__(pandas_frame)

    def qc_engine_switch_cost(self, other_qc_cls):
        return {
            CloudQC: QCCoercionCost.COST_MEDIUM,
            ClusterQC: QCCoercionCost.COST_ZERO,
            DefaultQC: None,  # cluster qc knows nothing about default qc
            LocalMachineQC: QCCoercionCost.COST_MEDIUM,
            PicoQC: QCCoercionCost.COST_HIGH,
        }[other_qc_cls]


class LocalMachineQC(NativeQueryCompiler):
    "Represents a local machine query compiler"

    def __init__(self, pandas_frame):
        self._modin_frame = pandas_frame
        super().__init__(pandas_frame)

    def qc_engine_switch_cost(self, other_qc_cls):
        return {
            CloudQC: QCCoercionCost.COST_MEDIUM,
            ClusterQC: QCCoercionCost.COST_LOW,
            LocalMachineQC: QCCoercionCost.COST_ZERO,
            PicoQC: QCCoercionCost.COST_MEDIUM,
        }[other_qc_cls]


class PicoQC(NativeQueryCompiler):
    "Represents a query compiler with very few resources"

    def __init__(self, pandas_frame):
        self._modin_frame = pandas_frame
        super().__init__(pandas_frame)

    def qc_engine_switch_cost(self, other_qc_cls):
        return {
            CloudQC: QCCoercionCost.COST_LOW,
            ClusterQC: QCCoercionCost.COST_LOW,
            LocalMachineQC: QCCoercionCost.COST_LOW,
            PicoQC: QCCoercionCost.COST_ZERO,
        }[other_qc_cls]


class AdversarialQC(NativeQueryCompiler):
    "Represents a query compiler which returns non-sensiscal costs"

    def __init__(self, pandas_frame):
        self._modin_frame = pandas_frame
        super().__init__(pandas_frame)

    def qc_engine_switch_cost(self, other_qc_cls):
        return {
            CloudQC: -1000,
            ClusterQC: 10000,
            AdversarialQC: QCCoercionCost.COST_ZERO,
        }[other_qc_cls]


class DefaultQC(NativeQueryCompiler):
    "Represents a query compiler with no costing information"

    def __init__(self, pandas_frame):
        self._modin_frame = pandas_frame
        super().__init__(pandas_frame)


class DefaultQC2(NativeQueryCompiler):
    "Represents a query compiler with no costing information, but different."

    def __init__(self, pandas_frame):
        self._modin_frame = pandas_frame
        super().__init__(pandas_frame)


@pytest.fixture()
def cloud_df():
    return CloudQC(pandas.DataFrame([0, 1, 2]))


@pytest.fixture()
def cluster_df():
    return ClusterQC(pandas.DataFrame([0, 1, 2]))


@pytest.fixture()
def local_df():
    return LocalMachineQC(pandas.DataFrame([0, 1, 2]))


@pytest.fixture()
def pico_df():
    return PicoQC(pandas.DataFrame([0, 1, 2]))


@pytest.fixture()
def adversarial_df():
    return AdversarialQC(pandas.DataFrame([0, 1, 2]))


@pytest.fixture()
def default_df():
    return DefaultQC(pandas.DataFrame([0, 1, 2]))


@pytest.fixture()
def default2_df():
    return DefaultQC(pandas.DataFrame([0, 1, 2]))


def test_two_same_qc_types_noop(pico_df):
    df3 = pico_df.concat(axis=1, other=pico_df)
    assert type(df3) is type(pico_df)


def test_two_two_qc_types_rhs(pico_df, cluster_df):
    df3 = pico_df.concat(axis=1, other=cluster_df)
    assert type(df3) is type(cluster_df)  # should move to cluster


def test_two_two_qc_types_lhs(pico_df, cluster_df):
    df3 = cluster_df.concat(axis=1, other=pico_df)
    assert type(df3) is type(cluster_df)  # should move to cluster


@pytest.mark.parametrize(
    "df1, df2, df3, df4, result_type",
    [
        # no-op
        ("cloud_df", "cloud_df", "cloud_df", "cloud_df", CloudQC),
        # moving all dfs to cloud is 1250, moving to cluster is 1000
        # regardless of how they are ordered
        ("pico_df", "local_df", "cluster_df", "cloud_df", ClusterQC),
        ("cloud_df", "local_df", "cluster_df", "pico_df", ClusterQC),
        ("cloud_df", "cluster_df", "local_df", "pico_df", ClusterQC),
        ("cloud_df", "cloud_df", "local_df", "pico_df", CloudQC),
        # Still move everything to cloud
        ("pico_df", "pico_df", "pico_df", "cloud_df", CloudQC),
    ],
)
def test_mixed_dfs(df1, df2, df3, df4, result_type, request):
    df1 = request.getfixturevalue(df1)
    df2 = request.getfixturevalue(df2)
    df3 = request.getfixturevalue(df3)
    df4 = request.getfixturevalue(df4)
    result = df1.concat(axis=1, other=[df2, df3, df4])
    assert type(result) is result_type


# This currently passes because we have no "max cost" associated
# with a particular QC, so we would move all data to the PicoQC
# As soon as we can represent "max-cost" the result of this operation
# should be to move all dfs to the CloudQC
def test_extreme_pico(pico_df, cloud_df):
    result = cloud_df.concat(
        axis=1, other=[pico_df, pico_df, pico_df, pico_df, pico_df, pico_df, pico_df]
    )
    assert type(result) is PicoQC


def test_call_on_non_qc(pico_df, cloud_df):
    pico_df1 = pd.DataFrame(query_compiler=pico_df)
    cloud_df1 = pd.DataFrame(query_compiler=cloud_df)

    df1 = pd.concat([pico_df1, cloud_df1])
    assert type(df1._query_compiler) is CloudQC


def test_adversarial_high(adversarial_df, cluster_df):
    with pytest.raises(ValueError):
        adversarial_df.concat(axis=1, other=cluster_df)


def test_adversarial_low(adversarial_df, cloud_df):
    with pytest.raises(ValueError):
        adversarial_df.concat(axis=1, other=cloud_df)


def test_two_two_qc_types_default_rhs(default_df, cluster_df):
    # none of the query compilers know about each other here
    # so we default to the caller
    df3 = default_df.concat(axis=1, other=cluster_df)
    assert type(df3) is type(default_df)  # should move to default


def test_two_two_qc_types_default_lhs(default_df, cluster_df):
    # none of the query compilers know about each other here
    # so we default to the caller
    df3 = cluster_df.concat(axis=1, other=default_df)
    assert type(df3) is type(cluster_df)  # should move to cluster


def test_two_two_qc_types_default_2_rhs(default_df, cloud_df):
    # cloud knows a bit about costing; so we prefer moving to there
    df3 = default_df.concat(axis=1, other=cloud_df)
    assert type(df3) is type(cloud_df)  # should move to cloud


def test_two_two_qc_types_default_2_lhs(default_df, cloud_df):
    # cloud knows a bit about costing; so we prefer moving to there
    df3 = cloud_df.concat(axis=1, other=default_df)
    assert type(df3) is type(cloud_df)  # should move to cloud


def test_default_to_caller(default_df, default2_df):
    # No qc knows anything; default to caller
    df3 = default_df.concat(axis=1, other=default2_df)
    assert type(df3) is type(default_df)  # should stay on caller
    df3 = default2_df.concat(axis=1, other=default_df)
    assert type(df3) is type(default2_df)  # should stay on caller
