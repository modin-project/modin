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
from modin.config.envvars import Backend, Engine, Execution
from modin.core.execution.dispatching.factories import factories
from modin.core.execution.dispatching.factories.factories import BaseFactory
from modin.core.io.io import BaseIO
from modin.core.storage_formats.base.query_compiler import QCCoercionCost
from modin.core.storage_formats.base.query_compiler_calculator import (
    BackendCostCalculator,
)
from modin.core.storage_formats.pandas.native_query_compiler import NativeQueryCompiler
from modin.tests.pandas.utils import df_equals


class CloudQC(NativeQueryCompiler):
    "Represents a cloud-hosted query compiler"

    def get_backend(self):
        return "Cloud"

    def qc_engine_switch_max_cost(self):
        return QCCoercionCost.COST_IMPOSSIBLE

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

    def get_backend(self):
        return "Cluster"

    def qc_engine_switch_max_cost(self):
        return QCCoercionCost.COST_HIGH

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

    def get_backend(self):
        return "Local_machine"

    def qc_engine_switch_max_cost(self):
        return QCCoercionCost.COST_MEDIUM

    def qc_engine_switch_cost(self, other_qc_cls):
        return {
            CloudQC: QCCoercionCost.COST_MEDIUM,
            ClusterQC: QCCoercionCost.COST_LOW,
            LocalMachineQC: QCCoercionCost.COST_ZERO,
            PicoQC: QCCoercionCost.COST_MEDIUM,
        }[other_qc_cls]


class PicoQC(NativeQueryCompiler):
    "Represents a query compiler with very few resources"

    def get_backend(self):
        return "Pico"

    def qc_engine_switch_max_cost(self):
        return QCCoercionCost.COST_LOW

    def qc_engine_switch_cost(self, other_qc_cls):
        return {
            CloudQC: QCCoercionCost.COST_LOW,
            ClusterQC: QCCoercionCost.COST_LOW,
            LocalMachineQC: QCCoercionCost.COST_LOW,
            PicoQC: QCCoercionCost.COST_ZERO,
        }[other_qc_cls]


class AdversarialQC(NativeQueryCompiler):
    "Represents a query compiler which returns non-sensical costs"

    def get_backend(self):
        return "Adversarial"

    def qc_engine_switch_cost(self, other_qc_cls):
        return {
            CloudQC: -1000,
            ClusterQC: 10000,
            AdversarialQC: QCCoercionCost.COST_ZERO,
        }[other_qc_cls]


class DefaultQC(NativeQueryCompiler):
    "Represents a query compiler with no costing information"

    def get_backend(self):
        return "Test_casting_default"


class DefaultQC2(NativeQueryCompiler):
    "Represents a query compiler with no costing information, but different."

    def get_backend(self):
        return "Test_casting_default_2"


def register_backend(name, qc):
    class TestCasterIO(BaseIO):
        _should_warn_on_default_to_pandas: bool = False
        query_compiler_cls = qc

    class TestCasterFactory(BaseFactory):
        @classmethod
        def prepare(cls):
            cls.io_cls = TestCasterIO

    factory_name = f"{name}OnNativeFactory"
    setattr(factories, factory_name, TestCasterFactory)
    Engine.add_option(name)
    Backend.register_backend(name, Execution(name, "Native"))


register_backend("Pico", PicoQC)
register_backend("Cluster", ClusterQC)
register_backend("Cloud", CloudQC)
register_backend("Local_machine", LocalMachineQC)
register_backend("Adversarial", AdversarialQC)
register_backend("Test_casting_default", DefaultQC)
register_backend("Test_casting_default_2", DefaultQC2)


@pytest.fixture()
def cloud_df():
    return pd.DataFrame(query_compiler=CloudQC(pandas.DataFrame([0, 1, 2])))


@pytest.fixture()
def cluster_df():
    return pd.DataFrame(query_compiler=ClusterQC(pandas.DataFrame([0, 1, 2])))


@pytest.fixture()
def local_df():
    return pd.DataFrame(query_compiler=LocalMachineQC(pandas.DataFrame([0, 1, 2])))


@pytest.fixture()
def pico_df():
    return pd.DataFrame(query_compiler=PicoQC(pandas.DataFrame([0, 1, 2])))


@pytest.fixture()
def adversarial_df():
    return pd.DataFrame(query_compiler=AdversarialQC(pandas.DataFrame([0, 1, 2])))


@pytest.fixture()
def default_df():
    return pd.DataFrame(query_compiler=DefaultQC(pandas.DataFrame([0, 1, 2])))


@pytest.fixture()
def default2_df():
    return pd.DataFrame(query_compiler=DefaultQC2(pandas.DataFrame([0, 1, 2])))


def test_two_same_backend(pico_df):
    df3 = pd.concat([pico_df, pico_df], axis=1)
    assert pico_df.get_backend() == "Pico"
    assert df3.get_backend() == "Pico"


def test_cast_to_second_backend_with_concat(pico_df, cluster_df):
    df3 = pd.concat([pico_df, cluster_df], axis=1)
    assert pico_df.get_backend() == "Pico"
    assert cluster_df.get_backend() == "Cluster"
    assert df3.get_backend() == "Cluster"  # result should be on cluster


def test_cast_to_second_backend_with___init__(pico_df, cluster_df):
    df3 = pd.DataFrame({"pico": pico_df.iloc[:, 0], "cluster": cluster_df.iloc[:, 0]})
    assert pico_df.get_backend() == "Pico"
    assert cluster_df.get_backend() == "Cluster"
    assert df3.get_backend() == "Cluster"  # result should be on cluster


def test_cast_to_first_backend(pico_df, cluster_df):
    df3 = pd.concat([cluster_df, pico_df], axis=1)
    assert pico_df.get_backend() == "Pico"
    assert cluster_df.get_backend() == "Cluster"
    assert df3.get_backend() == cluster_df.get_backend()  # result should be on cluster


def test_cast_to_first_backend_with___init__(pico_df, cluster_df):
    df3 = pd.DataFrame(
        {
            "cluster": cluster_df.iloc[:, 0],
            "pico": pico_df.iloc[:, 0],
        }
    )
    assert pico_df.get_backend() == "Pico"
    assert cluster_df.get_backend() == "Cluster"
    assert df3.get_backend() == "Cluster"  # result should be on cluster


def test_no_solution(pico_df, local_df, cluster_df, cloud_df):
    with pytest.raises(ValueError, match=r"Pico,Local_machine,Cluster,Cloud"):
        pd.concat(axis=1, objs=[pico_df, local_df, cluster_df, cloud_df])


@pytest.mark.parametrize(
    "df1, df2, df3, df4, expected_result_backend",
    [
        # no-op
        ("cloud_df", "cloud_df", "cloud_df", "cloud_df", "Cloud"),
        # moving all dfs to cloud is 1250, moving to cluster is 1000
        # regardless of how they are ordered
        ("pico_df", "local_df", "cluster_df", "cloud_df", None),
        ("cloud_df", "local_df", "cluster_df", "pico_df", None),
        ("cloud_df", "cluster_df", "local_df", "pico_df", None),
        ("cloud_df", "cloud_df", "local_df", "pico_df", "Cloud"),
        # Still move everything to cloud
        ("pico_df", "pico_df", "pico_df", "cloud_df", "Cloud"),
        ("pico_df", "pico_df", "local_df", "cloud_df", "Cloud"),
    ],
)
def test_mixed_dfs(df1, df2, df3, df4, expected_result_backend, request):
    df1 = request.getfixturevalue(df1)
    df2 = request.getfixturevalue(df2)
    df3 = request.getfixturevalue(df3)
    df4 = request.getfixturevalue(df4)
    if expected_result_backend is None:
        with pytest.raises(ValueError):
            pd.concat(axis=1, objs=[df1, df2, df3, df4])
    else:
        result = pd.concat(axis=1, objs=[df1, df2, df3, df4])
        assert result.get_backend() == expected_result_backend


def test_adversarial_high(adversarial_df, cluster_df):
    with pytest.raises(ValueError):
        pd.concat([adversarial_df, cluster_df], axis=1)


def test_adversarial_low(adversarial_df, cloud_df):
    with pytest.raises(ValueError):
        pd.concat([adversarial_df, cloud_df], axis=1)


def test_two_two_qc_types_default_rhs(default_df, cluster_df):
    # none of the query compilers know about each other here
    # so we default to the caller
    df3 = pd.concat([default_df, cluster_df], axis=1)
    assert default_df.get_backend() == "Test_casting_default"
    assert cluster_df.get_backend() == "Cluster"
    assert df3.get_backend() == default_df.get_backend()  # should move to default


def test_two_two_qc_types_default_lhs(default_df, cluster_df):
    # none of the query compilers know about each other here
    # so we default to the caller
    df3 = pd.concat([cluster_df, default_df], axis=1)
    assert default_df.get_backend() == "Test_casting_default"
    assert cluster_df.get_backend() == "Cluster"
    assert df3.get_backend() == cluster_df.get_backend()  # should move to cluster


def test_two_two_qc_types_default_2_rhs(default_df, cloud_df):
    # cloud knows a bit about costing; so we prefer moving to there
    df3 = pd.concat([default_df, cloud_df], axis=1)
    assert default_df.get_backend() == "Test_casting_default"
    assert cloud_df.get_backend() == "Cloud"
    assert df3.get_backend() == cloud_df.get_backend()  # should move to cloud


def test_two_two_qc_types_default_2_lhs(default_df, cloud_df):
    # cloud knows a bit about costing; so we prefer moving to there
    df3 = pd.concat([cloud_df, default_df], axis=1)
    assert default_df.get_backend() == "Test_casting_default"
    assert cloud_df.get_backend() == "Cloud"
    assert df3.get_backend() == cloud_df.get_backend()  # should move to cloud


def test_default_to_caller(default_df, default2_df):
    # No qc knows anything; default to caller

    df3 = pd.concat([default_df, default2_df], axis=1)
    assert df3.get_backend() == default_df.get_backend()  # should stay on caller

    df3 = pd.concat([default2_df, default_df], axis=1)
    assert df3.get_backend() == default2_df.get_backend()  # should stay on caller

    df3 = pd.concat([default_df, default_df], axis=1)
    assert df3.get_backend() == default_df.get_backend()  # no change


def test_no_qc_to_calculate():
    calculator = BackendCostCalculator()
    with pytest.raises(ValueError):
        calculator.calculate()


def test_qc_default_self_cost(default_df, default2_df):
    assert (
        default_df._query_compiler.qc_engine_switch_cost(
            type(default2_df._query_compiler)
        )
        is None
    )
    assert (
        default_df._query_compiler.qc_engine_switch_cost(
            type(default_df._query_compiler)
        )
        is QCCoercionCost.COST_ZERO
    )


def test_qc_casting_changed_operation(pico_df, cloud_df):
    pico_df1 = pico_df
    cloud_df1 = cloud_df
    native_cdf2 = cloud_df1._to_pandas()
    native_pdf2 = pico_df1._to_pandas()
    expected = native_cdf2 + native_pdf2
    # test both directions
    df_cast_to_rhs = pico_df1 + cloud_df1
    df_cast_to_lhs = cloud_df1 + pico_df1
    assert df_cast_to_rhs._to_pandas().equals(expected)
    assert df_cast_to_lhs._to_pandas().equals(expected)


def test_qc_mixed_loc(pico_df, cloud_df):
    pico_df1 = pico_df
    cloud_df1 = cloud_df
    assert pico_df1[pico_df1[0][0]][cloud_df1[0][1]] == 1
    assert pico_df1[cloud_df1[0][0]][pico_df1[0][1]] == 1
    assert cloud_df1[pico_df1[0][0]][pico_df1[0][1]] == 1


def test_setitem_in_place_with_self_switching_backend(cloud_df, local_df):
    local_df.iloc[1, 0] = cloud_df.iloc[1, 0] + local_df.iloc[1, 0]
    # compute happens in cloud, but we have to make sure that we propagate the
    # in-place update to the local_df
    df_equals(
        local_df,
        pandas.DataFrame(
            [
                0,
                2,
                2,
            ]
        ),
    )
    assert local_df.get_backend() == "Local_machine"
    assert cloud_df.get_backend() == "Cloud"
