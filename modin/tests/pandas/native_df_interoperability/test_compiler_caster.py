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

import contextlib
import json
import logging
from io import StringIO
from types import MappingProxyType
from typing import Iterator
from unittest import mock

import pandas
import pytest
from pytest import param

import modin.pandas as pd
from modin.config import context as config_context
from modin.config.envvars import (
    Backend,
    Engine,
    Execution,
    NativePandasMaxRows,
    NativePandasTransferThreshold,
)
from modin.core.execution.dispatching.factories import factories
from modin.core.execution.dispatching.factories.factories import BaseFactory
from modin.core.io.io import BaseIO
from modin.core.storage_formats.base.query_compiler import QCCoercionCost
from modin.core.storage_formats.base.query_compiler_calculator import (
    BackendCostCalculator,
)
from modin.core.storage_formats.pandas.native_query_compiler import NativeQueryCompiler
from modin.core.storage_formats.pandas.query_compiler_caster import (
    _GENERAL_EXTENSIONS,
    register_function_for_post_op_switch,
    register_function_for_pre_op_switch,
)
from modin.logging import DEFAULT_LOGGER_NAME
from modin.logging.metrics import add_metric_handler, clear_metric_handler
from modin.pandas.api.extensions import register_pd_accessor
from modin.tests.pandas.utils import (
    create_test_dfs,
    default_to_pandas_ignore_string,
    df_equals,
    eval_general,
)

# Some modin methods warn about defaulting to pandas at the API layer. That's
# expected and not an error as it would be normally.
pytestmark = pytest.mark.filterwarnings(default_to_pandas_ignore_string)

BIG_DATA_CLOUD_MIN_NUM_ROWS = 10
SMALL_DATA_NUM_ROWS = 5


class CalculatorTestQc(NativeQueryCompiler):
    """
    A subclass of NativeQueryCompiler with simpler cost functions.

    We MAY eventually want to stop overriding the superclass's cost functions.
    """

    @classmethod
    def move_to_me_cost(cls, other_qc, api_cls_name, operation, arguments):
        if isinstance(other_qc, cls):
            return QCCoercionCost.COST_ZERO
        return None

    def stay_cost(self, api_cls_name, operation, arguments):
        return QCCoercionCost.COST_ZERO

    def move_to_cost(self, other_qc_type, api_cls_name, operation, arguments):
        if isinstance(self, other_qc_type):
            return QCCoercionCost.COST_ZERO
        return None


class CloudQC(CalculatorTestQc):
    "Represents a cloud-hosted query compiler"

    def get_backend(self):
        return "Cloud"

    def max_cost(self):
        return QCCoercionCost.COST_IMPOSSIBLE

    def move_to_cost(self, other_qc_cls, api_cls_name, op, arguments):
        assert op is not None
        assert api_cls_name in [
            None,
            "_iLocIndexer",
            "_LocationIndexerBase",
            "Series",
            "DataFrame",
            "BasePandasDataset",
        ]
        return {
            CloudQC: QCCoercionCost.COST_ZERO,
            CloudQCHighSelf: QCCoercionCost.COST_LOW,
            ClusterQC: QCCoercionCost.COST_MEDIUM,
            DefaultQC: QCCoercionCost.COST_MEDIUM,
            LocalMachineQC: QCCoercionCost.COST_HIGH,
            PicoQC: QCCoercionCost.COST_IMPOSSIBLE,
            OmniscientEagerQC: None,
            OmniscientLazyQC: None,
        }[other_qc_cls]

    def stay_cost(self, api_cls_name, op, arguments):
        return QCCoercionCost.COST_ZERO


class CloudQCHighSelf(CloudQC):
    def get_backend(self):
        return "Cloud_High_Self"

    def stay_cost(self, api_cls_name, op, arguments):
        return QCCoercionCost.COST_HIGH


class ClusterQC(CalculatorTestQc):
    "Represents a local network cluster query compiler"

    def get_backend(self):
        return "Cluster"

    def max_cost(self):
        return QCCoercionCost.COST_HIGH

    def move_to_cost(self, other_qc_cls, api_cls_name, op, arguments):
        return {
            CloudQC: QCCoercionCost.COST_MEDIUM,
            CloudQCHighSelf: QCCoercionCost.COST_MEDIUM,
            ClusterQC: QCCoercionCost.COST_ZERO,
            DefaultQC: None,  # cluster qc knows nothing about default qc
            LocalMachineQC: QCCoercionCost.COST_MEDIUM,
            PicoQC: QCCoercionCost.COST_HIGH,
        }[other_qc_cls]


class LocalMachineQC(CalculatorTestQc):
    "Represents a local machine query compiler"

    def get_backend(self):
        return "Local_machine"

    def max_cost(self):
        return QCCoercionCost.COST_MEDIUM

    def move_to_cost(self, other_qc_cls, api_cls_name, op, arguments):
        return {
            CloudQC: QCCoercionCost.COST_MEDIUM,
            CloudQCHighSelf: QCCoercionCost.COST_MEDIUM,
            ClusterQC: QCCoercionCost.COST_LOW,
            LocalMachineQC: QCCoercionCost.COST_ZERO,
            PicoQC: QCCoercionCost.COST_MEDIUM,
        }[other_qc_cls]


class PicoQC(CalculatorTestQc):
    "Represents a query compiler with very few resources"

    def get_backend(self):
        return "Pico"

    def max_cost(self):
        return QCCoercionCost.COST_LOW

    def move_to_cost(self, other_qc_cls, api_cls_name, op, arguments):
        return {
            CloudQC: QCCoercionCost.COST_LOW,
            CloudQCHighSelf: QCCoercionCost.COST_LOW,
            ClusterQC: QCCoercionCost.COST_LOW,
            LocalMachineQC: QCCoercionCost.COST_LOW,
            PicoQC: QCCoercionCost.COST_ZERO,
        }[other_qc_cls]


class AdversarialQC(CalculatorTestQc):
    "Represents a query compiler which returns non-sensical costs"

    def get_backend(self):
        return "Adversarial"

    def move_to_cost(self, other_qc_cls, api_cls_name, op, arguments):
        return {
            CloudQC: -1000,
            CloudQCHighSelf: -1000,
            ClusterQC: 10000,
            AdversarialQC: QCCoercionCost.COST_ZERO,
        }[other_qc_cls]


class OmniscientEagerQC(CalculatorTestQc):
    "Represents a query compiler which knows a lot, and wants to steal work"

    def get_backend(self):
        return "Eager"

    # keep other workloads from getting my workload
    def move_to_cost(self, other_qc_cls, api_cls_name, op, arguments):
        if OmniscientEagerQC is other_qc_cls:
            return QCCoercionCost.COST_ZERO
        return QCCoercionCost.COST_IMPOSSIBLE

    # try to force other workloads to my engine
    @classmethod
    def move_to_me_cost(cls, other_qc, api_cls_name, operation, arguments):
        return QCCoercionCost.COST_ZERO


class OmniscientLazyQC(CalculatorTestQc):
    "Represents a query compiler which knows a lot, and wants to avoid work"

    def get_backend(self):
        return "Lazy"

    # encorage other engines to take my workload
    def move_to_cost(self, other_qc_cls, api_cls_name, op, arguments):
        return QCCoercionCost.COST_ZERO

    # try to keep other workloads from getting my workload
    @classmethod
    def move_to_me_cost(cls, other_qc, api_cls_name, operation, arguments):
        if isinstance(other_qc, cls):
            return QCCoercionCost.COST_ZERO
        return QCCoercionCost.COST_IMPOSSIBLE


class DefaultQC(CalculatorTestQc):
    "Represents a query compiler with no costing information"

    def get_backend(self):
        return "Test_casting_default"


class DefaultQC2(CalculatorTestQc):
    "Represents a query compiler with no costing information, but different."

    def get_backend(self):
        return "Test_casting_default_2"


class BaseTestAutoMover(NativeQueryCompiler):

    _MAX_SIZE_THIS_ENGINE_CAN_HANDLE = BIG_DATA_CLOUD_MIN_NUM_ROWS

    def __init__(self, pandas_frame):
        super().__init__(pandas_frame)


class CloudForBigDataQC(BaseTestAutoMover):
    """Represents a cloud-hosted query compiler that prefers to stay on the cloud only for big data"""

    # Operations are more costly on this engine, even though it can handle larger datasets
    _MAX_SIZE_THIS_ENGINE_CAN_HANDLE = BIG_DATA_CLOUD_MIN_NUM_ROWS * 10
    _OPERATION_INITIALIZATION_OVERHEAD = QCCoercionCost.COST_MEDIUM
    _OPERATION_PER_ROW_OVERHEAD = 10

    def __init__(self, pandas_frame):
        super().__init__(pandas_frame)

    def stay_cost(self, api_cls_name, operation, arguments):
        if operation == "read_json":
            return QCCoercionCost.COST_IMPOSSIBLE
        return super().stay_cost(api_cls_name, operation, arguments)

    def get_backend(self) -> str:
        return "Big_Data_Cloud"

    def max_cost(self):
        return QCCoercionCost.COST_IMPOSSIBLE * 10

    @classmethod
    def move_to_me_cost(cls, other_qc, api_cls_name, operation, arguments):
        if api_cls_name in ("DataFrame", "Series") and operation == "__init__":
            if (query_compiler := arguments.get("query_compiler")) is not None:
                # When we create a dataframe or series with a query compiler
                # input, we should not switch the resulting dataframe or series
                # to a different backend.
                return (
                    QCCoercionCost.COST_ZERO
                    if isinstance(query_compiler, cls)
                    else QCCoercionCost.COST_IMPOSSIBLE
                )
            else:
                # Moving the in-memory __init__ inputs to the cloud is expensive.
                return QCCoercionCost.COST_HIGH
        return super().move_to_me_cost(
            cls, other_qc, api_cls_name, operation, arguments
        )


class LocalForSmallDataQC(BaseTestAutoMover):
    """Represents a local query compiler that prefers small data."""

    # Operations are cheap on this engine for small data, but there is an upper bound
    _MAX_SIZE_THIS_ENGINE_CAN_HANDLE = BIG_DATA_CLOUD_MIN_NUM_ROWS
    _OPERATION_PER_ROW_OVERHEAD = 1

    def __init__(self, pandas_frame):
        super().__init__(pandas_frame)

    def get_backend(self) -> str:
        return "Small_Data_Local"

    def max_cost(self):
        return QCCoercionCost.COST_IMPOSSIBLE * 10


def register_backend(name, qc):
    class TestCasterIO(BaseIO):
        _should_warn_on_default_to_pandas: bool = False
        query_compiler_cls = qc

    class TestCasterFactory(BaseFactory):
        @classmethod
        def prepare(cls):
            cls.io_cls = TestCasterIO

    TestCasterFactory.prepare()

    factory_name = f"{name}OnNativeFactory"
    setattr(factories, factory_name, TestCasterFactory)
    Engine.add_option(name)
    Backend.register_backend(name, Execution(name, "Native"))


register_backend("Pico", PicoQC)
register_backend("Cluster", ClusterQC)
register_backend("Cloud", CloudQC)
register_backend("Cloud_High_Self", CloudQCHighSelf)
register_backend("Local_Machine", LocalMachineQC)
register_backend("Adversarial", AdversarialQC)
register_backend("Eager", OmniscientEagerQC)
register_backend("Lazy", OmniscientLazyQC)
register_backend("Test_Casting_Default", DefaultQC)
register_backend("Test_Casting_Default_2", DefaultQC2)
register_backend("Big_Data_Cloud", CloudForBigDataQC)
register_backend("Small_Data_Local", LocalForSmallDataQC)


@pytest.fixture()
def cloud_df():
    return pd.DataFrame(query_compiler=CloudQC(pandas.DataFrame([0, 1, 2])))


@pytest.fixture()
def cloud_high_self_df():
    return pd.DataFrame(query_compiler=CloudQCHighSelf(pandas.DataFrame([0, 1, 2])))


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
def eager_df():
    return pd.DataFrame(query_compiler=OmniscientEagerQC(pandas.DataFrame([0, 1, 2])))


@pytest.fixture()
def lazy_df():
    return pd.DataFrame(query_compiler=OmniscientLazyQC(pandas.DataFrame([0, 1, 2])))


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


def test_cast_to_second_backend_with_concat(pico_df, cluster_df, caplog):
    with caplog.at_level(level=logging.INFO, logger=DEFAULT_LOGGER_NAME):
        df3 = pd.concat([pico_df, cluster_df], axis=1)
    assert pico_df.get_backend() == "Pico"
    assert cluster_df.get_backend() == "Cluster"
    assert df3.get_backend() == "Cluster"  # result should be on cluster

    log_records = caplog.records
    assert len(log_records) == 1
    assert log_records[0].name == DEFAULT_LOGGER_NAME
    assert log_records[0].levelno == logging.INFO
    assert log_records[0].message.startswith("BackendCostCalculator Results: ")


def test_cast_to_second_backend_with_concat_uses_second_backend_api_override(
    pico_df, cluster_df
):
    register_pd_accessor(name="concat", backend="Cluster")(
        lambda *args, **kwargs: "custom_concat_result"
    )
    assert pd.concat([pico_df, cluster_df], axis=1) == "custom_concat_result"
    assert pico_df.get_backend() == "Pico"
    assert cluster_df.get_backend() == "Cluster"


def test_moving_pico_to_cluster_in_place_calls_set_backend_only_once_github_issue_7490(
    pico_df, cluster_df
):
    with mock.patch.object(
        pd.DataFrame, "set_backend", wraps=pico_df.set_backend
    ) as mock_set_backend:
        pico_df.set_backend(cluster_df.get_backend(), inplace=True)
    assert pico_df.get_backend() == "Cluster"
    mock_set_backend.assert_called_once_with("Cluster", inplace=True)


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


def test_cast_to_first_backend_with_concat_uses_first_backend_api_override(
    pico_df, cluster_df
):
    register_pd_accessor(name="concat", backend="Cluster")(
        lambda *args, **kwargs: "custom_concat_result"
    )
    assert pd.concat([cluster_df, pico_df], axis=1) == "custom_concat_result"
    assert pico_df.get_backend() == "Pico"
    assert cluster_df.get_backend() == "Cluster"


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


def test_self_cost_causes_move(cloud_high_self_df, cluster_df):
    """
    Test that ``self_cost`` is being properly considered.

    Cost to stay on cloud_high_self is HIGH, but moving to cluster is MEDIUM.
    Cost to stay on cluster is ZERO, and moving to cloud_high_self is MEDIUM.

    With two dataframes, one on each backend, the total cost of using
    ``cloud_high_self`` as the final backend is:
    ``stay_cost(cloud_high_self) + move_cost(cluster->cloud_high_self)``
    which is ``HIGH + MEDIUM``.
    The total cost of using ``cluster`` as the final backend is:
    ``stay_cost(cluster) + move_cost(cloud_high_self->cluster)``
    which is ``ZERO + MEDIUM``.

    So we should select ``cluster``.
    """
    result = pd.concat([cloud_high_self_df, cluster_df])
    assert result.get_backend() == "Cluster"

    result = pd.concat([cluster_df, cloud_high_self_df])
    assert result.get_backend() == "Cluster"


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
    calculator = BackendCostCalculator(
        operation_arguments=MappingProxyType({}),
        api_cls_name=None,
        operation="operation0",
    )
    with pytest.raises(ValueError):
        calculator.calculate()


def test_qc_default_self_cost(default_df, default2_df):
    assert (
        default_df._query_compiler.move_to_cost(
            other_qc_type=type(default2_df._query_compiler),
            api_cls_name=None,
            operation="operation0",
            arguments=MappingProxyType({}),
        )
        is None
    )
    assert (
        default_df._query_compiler.move_to_cost(
            other_qc_type=type(default_df._query_compiler),
            api_cls_name=None,
            operation="operation0",
            arguments=MappingProxyType({}),
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


def test_information_asymmetry(default_df, cloud_df, eager_df, lazy_df):
    # normally, the default query compiler should be chosen
    # here, but since eager knows about default, but not
    # the other way around, eager has a special ability to
    # control the directionality of the cast.
    df = default_df.merge(eager_df)
    assert type(df) is type(eager_df)
    df = cloud_df.merge(eager_df)
    assert type(df) is type(eager_df)

    # lazy_df tries to pawn off work on other engines
    df = default_df.merge(lazy_df)
    assert type(df) is type(default_df)
    df = cloud_df.merge(lazy_df)
    assert type(df) is type(cloud_df)


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


@pytest.mark.parametrize("pin_local", [True, False], ids=["pinned", "unpinned"])
def test_switch_local_to_cloud_with_iloc___setitem__(local_df, cloud_df, pin_local):
    if pin_local:
        local_df = local_df.pin_backend()
    local_df.iloc[:, 0] = cloud_df.iloc[:, 0] + 1
    expected_pandas = local_df._to_pandas()
    expected_pandas.iloc[:, 0] = cloud_df._to_pandas().iloc[:, 0] + 1
    df_equals(local_df, expected_pandas)
    assert local_df.get_backend() == "Local_machine" if pin_local else "Cloud"


def test_stay_or_move_evaluation(cloud_high_self_df, default_df):
    default_cls = type(default_df._get_query_compiler())
    cloud_cls = type(cloud_high_self_df._get_query_compiler())
    empty_arguments = MappingProxyType({})

    stay_cost = cloud_high_self_df._get_query_compiler().stay_cost(
        "Series", "myop", arguments=empty_arguments
    )
    move_cost = cloud_high_self_df._get_query_compiler().move_to_cost(
        default_cls, "Series", "myop", arguments=empty_arguments
    )
    if stay_cost > move_cost:
        df = cloud_high_self_df.move_to("Test_casting_default")
    else:
        assert False

    stay_cost = df._get_query_compiler().stay_cost(
        "Series", "myop", arguments=empty_arguments
    )
    move_cost = df._get_query_compiler().move_to_cost(
        cloud_cls, "Series", "myop", arguments=empty_arguments
    )
    assert stay_cost is not None
    assert move_cost is None


def test_max_shape(cloud_df):
    # default implementation matches df.shape
    assert cloud_df.shape == cloud_df._query_compiler._max_shape()


@contextlib.contextmanager
def backend_test_context(test_backend: str, choices: set) -> Iterator[None]:

    old_default_backend = Backend.get()
    old_backend_choices = Backend.get_active_backends()
    try:
        Backend.set_active_backends(choices)
        Backend.put(test_backend)
        yield
    finally:
        Backend.set_active_backends(old_backend_choices)
        Backend.put(old_default_backend)


class TestSwitchBackendPostOpDependingOnDataSize:
    def test_read_json(self):
        with backend_test_context(
            test_backend="Big_Data_Cloud",
            choices=("Big_Data_Cloud", "Small_Data_Local"),
        ):
            big_json = json.dumps({"col0": list(range(BIG_DATA_CLOUD_MIN_NUM_ROWS))})
            small_json = json.dumps(
                {"col0": list(range(BIG_DATA_CLOUD_MIN_NUM_ROWS - 1))}
            )
            assert pd.read_json(StringIO(big_json)).get_backend() == "Big_Data_Cloud"
            assert pd.read_json(StringIO(small_json)).get_backend() == "Big_Data_Cloud"
            register_function_for_post_op_switch(
                class_name=None, backend="Big_Data_Cloud", method="read_json"
            )
            assert pd.read_json(StringIO(big_json)).get_backend() == "Big_Data_Cloud"
            assert (
                pd.read_json(StringIO(small_json)).get_backend() == "Small_Data_Local"
            )

    @backend_test_context(
        test_backend="Big_Data_Cloud",
        choices=("Big_Data_Cloud", "Small_Data_Local"),
    )
    def test_read_json_logging_for_post_op_switch(self, caplog):
        register_function_for_post_op_switch(
            class_name=None, backend="Big_Data_Cloud", method="read_json"
        )
        with caplog.at_level(level=logging.INFO, logger=DEFAULT_LOGGER_NAME):
            assert (
                pd.read_json(
                    StringIO(
                        json.dumps(
                            {"col0": list(range(BIG_DATA_CLOUD_MIN_NUM_ROWS - 1))}
                        )
                    )
                ).get_backend()
                == "Small_Data_Local"
            )
        log_records = caplog.records
        assert len(log_records) == 2

        assert log_records[0].name == DEFAULT_LOGGER_NAME
        assert log_records[0].levelno == logging.INFO
        assert log_records[0].message.startswith(
            "After modin.pandas function read_json, considered moving to backend Small_Data_Local with"
        )

        assert log_records[1].name == DEFAULT_LOGGER_NAME
        assert log_records[1].levelno == logging.INFO
        assert log_records[1].message.startswith(
            "Chose to move to backend Small_Data_Local"
        )

    @backend_test_context(
        test_backend="Big_Data_Cloud",
        choices=("Big_Data_Cloud", "Small_Data_Local"),
    )
    def test_read_json_logging_for_post_op_not_switch(self, caplog):
        register_function_for_post_op_switch(
            class_name=None, backend="Big_Data_Cloud", method="read_json"
        )
        with caplog.at_level(level=logging.INFO, logger=DEFAULT_LOGGER_NAME):
            assert (
                pd.read_json(
                    StringIO(
                        json.dumps({"col0": list(range(BIG_DATA_CLOUD_MIN_NUM_ROWS))})
                    )
                ).get_backend()
                == "Big_Data_Cloud"
            )
        log_records = caplog.records
        assert len(log_records) == 2

        assert log_records[0].name == DEFAULT_LOGGER_NAME
        assert log_records[0].levelno == logging.INFO
        assert log_records[0].message.startswith(
            "After modin.pandas function read_json, considered moving to backend Small_Data_Local with"
        )

        assert log_records[1].name == DEFAULT_LOGGER_NAME
        assert log_records[1].levelno == logging.INFO
        assert log_records[1].message.startswith(
            "Chose not to switch backends after operation read_json"
        )

    @backend_test_context(
        test_backend="Big_Data_Cloud",
        choices=("Big_Data_Cloud", "Small_Data_Local"),
    )
    def test_progress_bar_shows_modin_pandas_for_general_functions(self):
        """Test that progress bar messages show 'modin.pandas.read_json' instead of 'None.read_json' for general functions."""
        with mock.patch("tqdm.auto.trange") as mock_trange:
            mock_trange.return_value = range(2)

            # Register a post-op switch for read_json (general function with class_name=None)
            register_function_for_post_op_switch(
                class_name=None, backend="Big_Data_Cloud", method="read_json"
            )

            # Create a small dataset that will trigger backend switch and show progress bar
            json_input = json.dumps(
                {"col0": list(range(BIG_DATA_CLOUD_MIN_NUM_ROWS - 1))}
            )

            # This should trigger a backend switch and show progress bar
            result_df = pd.read_json(StringIO(json_input))
            assert result_df.get_backend() == "Small_Data_Local"

            # Verify that trange was called with correct progress bar message
            mock_trange.assert_called_once()
            call_args = mock_trange.call_args
            desc = call_args[1]["desc"]  # Get the 'desc' keyword argument

            assert desc.startswith(
                "Transferring data from Big_Data_Cloud to Small_Data_Local for 'modin.pandas.read_json'"
            )

    def test_agg(self):
        with backend_test_context(
            test_backend="Big_Data_Cloud",
            choices=("Big_Data_Cloud", "Small_Data_Local"),
        ):
            df = pd.DataFrame([[1, 2], [3, 4]])
            assert df.get_backend() == "Big_Data_Cloud"
            assert df.sum().get_backend() == "Big_Data_Cloud"
            register_function_for_post_op_switch(
                class_name="DataFrame", backend="Big_Data_Cloud", method="sum"
            )
            assert df.get_backend() == "Big_Data_Cloud"
            assert df.sum().get_backend() == "Small_Data_Local"

    def test_agg_pinned(self):
        # The operation in test_agg would naturally cause an automatic switch, but the
        # absence of AutoSwitchBackend or the presence of a pin on the frame prevent this
        # switch from happening.
        with backend_test_context(
            test_backend="Big_Data_Cloud",
            choices=("Big_Data_Cloud", "Small_Data_Local"),
        ):
            register_function_for_post_op_switch(
                class_name="DataFrame", backend="Big_Data_Cloud", method="sum"
            )
            # No pin or config, should switch
            df = pd.DataFrame([[1, 2], [3, 4]])
            assert df.get_backend() == "Big_Data_Cloud"
            assert df.sum().get_backend() == "Small_Data_Local"
            # config set to false, should not switch
            with config_context(AutoSwitchBackend=False):
                df = pd.DataFrame([[1, 2], [3, 4]])
                assert df.get_backend() == "Big_Data_Cloud"
                assert df.sum().get_backend() == "Big_Data_Cloud"
            # no config, but data is pinned
            df = pd.DataFrame([[1, 2], [3, 4]]).pin_backend()
            assert df.get_backend() == "Big_Data_Cloud"
            assert df.sum().get_backend() == "Big_Data_Cloud"
            # a frame-level pin remains valid across a transformation
            df_copy = df + 1
            assert df_copy.get_backend() == "Big_Data_Cloud"
            assert df_copy.sum().get_backend() == "Big_Data_Cloud"
            # unpinning df allows a switch again
            df = df.unpin_backend()
            assert df.get_backend() == "Big_Data_Cloud"
            assert df.sum().get_backend() == "Small_Data_Local"
            df_copy = df + 1
            assert df_copy.get_backend() == "Big_Data_Cloud"
            assert df_copy.sum().get_backend() == "Small_Data_Local"
            # check in-place pin/unpin operations
            df.pin_backend(inplace=True)
            assert df.get_backend() == "Big_Data_Cloud"
            assert df.sum().get_backend() == "Big_Data_Cloud"
            df.unpin_backend(inplace=True)
            assert df.get_backend() == "Big_Data_Cloud"
            assert df.sum().get_backend() == "Small_Data_Local"

    @pytest.mark.parametrize(
        "num_groups, expected_backend",
        [
            (BIG_DATA_CLOUD_MIN_NUM_ROWS - 1, "Small_Data_Local"),
            (BIG_DATA_CLOUD_MIN_NUM_ROWS, "Big_Data_Cloud"),
        ],
    )
    @pytest.mark.parametrize(
        "groupby_class,operation",
        [
            param(
                "DataFrameGroupBy",
                lambda df: df.groupby("col0").sum(),
                id="DataFrameGroupBy",
            ),
            param(
                "SeriesGroupBy",
                lambda df: df.groupby("col0")["col1"].sum(),
                id="SeriesGroupBy",
            ),
        ],
    )
    def test_dataframe_groupby_agg_switches_for_small_result(
        self, num_groups, expected_backend, operation, groupby_class
    ):
        with backend_test_context(
            test_backend="Big_Data_Cloud",
            choices=("Big_Data_Cloud", "Small_Data_Local"),
        ):
            modin_df, pandas_df = create_test_dfs(
                {
                    "col0": list(range(num_groups)),
                    "col1": list(range(1, num_groups + 1)),
                }
            )

            assert modin_df.get_backend() == "Big_Data_Cloud"
            assert operation(modin_df).get_backend() == "Big_Data_Cloud"

            register_function_for_post_op_switch(
                class_name=groupby_class, backend="Big_Data_Cloud", method="sum"
            )

            assert modin_df.get_backend() == "Big_Data_Cloud"
            modin_result = operation(modin_df)
            pandas_result = operation(pandas_df)
            df_equals(modin_result, pandas_result)
            assert modin_result.get_backend() == expected_backend
            assert modin_df.get_backend() == "Big_Data_Cloud"

    @pytest.mark.parametrize(
        "groupby_class,operation",
        [
            param(
                "DataFrameGroupBy",
                lambda groupby: groupby.sum(),
                id="DataFrameGroupBy",
            ),
            param(
                "SeriesGroupBy",
                lambda groupby: groupby["col1"].sum(),
                id="SeriesGroupBy",
            ),
        ],
    )
    @pytest.mark.parametrize(
        "auto_switch_backend",
        [True, False],
        ids=lambda param: f"auto_switch_backend_{param}",
    )
    def test_auto_switch_config_can_disable_groupby_agg_auto_switch(
        self,
        operation,
        groupby_class,
        auto_switch_backend,
    ):
        num_groups = BIG_DATA_CLOUD_MIN_NUM_ROWS - 1
        with backend_test_context(
            test_backend="Big_Data_Cloud",
            choices=("Big_Data_Cloud", "Small_Data_Local"),
        ), config_context(AutoSwitchBackend=auto_switch_backend):
            modin_groupby, pandas_groupby = (
                df.groupby("col0")
                for df in create_test_dfs(
                    {
                        "col0": list(range(num_groups)),
                        "col1": list(range(1, num_groups + 1)),
                    }
                )
            )

            assert modin_groupby.get_backend() == "Big_Data_Cloud"
            assert operation(modin_groupby).get_backend() == "Big_Data_Cloud"

            register_function_for_post_op_switch(
                class_name=groupby_class, backend="Big_Data_Cloud", method="sum"
            )

            assert modin_groupby.get_backend() == "Big_Data_Cloud"
            modin_result = operation(modin_groupby)
            pandas_result = operation(pandas_groupby)
            df_equals(modin_result, pandas_result)
            assert modin_result.get_backend() == (
                "Small_Data_Local" if auto_switch_backend else "Big_Data_Cloud"
            )
            assert modin_groupby.get_backend() == "Big_Data_Cloud"

    @pytest.mark.parametrize(
        "groupby_class,groupby_operation,agg_operation",
        [
            param(
                "DataFrameGroupBy",
                lambda df: df.groupby("col0"),
                lambda groupby: groupby.sum(),
                id="DataFrameGroupBy",
            ),
            param(
                "SeriesGroupBy",
                lambda df: df.groupby("col0")["col1"],
                lambda groupby: groupby.sum(),
                id="SeriesGroupBy",
            ),
        ],
    )
    @backend_test_context(
        test_backend="Big_Data_Cloud",
        choices=("Big_Data_Cloud", "Small_Data_Local"),
    )
    def test_pinned_dataframe_prevents_groupby_backend_switch(
        self, groupby_class, groupby_operation, agg_operation
    ):
        """Test that pinning a DataFrame prevents groupby operations from switching backends."""
        modin_df, pandas_df = create_test_dfs(
            {
                "col0": list(range(BIG_DATA_CLOUD_MIN_NUM_ROWS - 1)),
                "col1": list(range(1, BIG_DATA_CLOUD_MIN_NUM_ROWS)),
            }
        )

        assert modin_df.get_backend() == "Big_Data_Cloud"

        # Pin the DataFrame
        modin_df.pin_backend(inplace=True)
        assert modin_df.is_backend_pinned()

        # Create groupby object - should inherit pin status from dataframe
        modin_groupby = groupby_operation(modin_df)
        pandas_groupby = groupby_operation(pandas_df)
        assert modin_groupby.is_backend_pinned()  # Inherited from DataFrame

        # Register a post-op switch that would normally move to Small_Data_Local
        register_function_for_post_op_switch(
            class_name=groupby_class, backend="Big_Data_Cloud", method="sum"
        )

        # The operation should stay on Big_Data_Cloud due to inherited pinning
        modin_result = agg_operation(modin_groupby)
        pandas_result = agg_operation(pandas_groupby)
        df_equals(modin_result, pandas_result)
        assert modin_result.get_backend() == "Big_Data_Cloud"

    @pytest.mark.parametrize(
        "groupby_class,groupby_operation,agg_operation",
        [
            param(
                "DataFrameGroupBy",
                lambda df: df.groupby("col0"),
                lambda groupby: groupby.sum(),
                id="DataFrameGroupBy",
            ),
            param(
                "SeriesGroupBy",
                lambda df: df.groupby("col0")["col1"],
                lambda groupby: groupby.sum(),
                id="SeriesGroupBy",
            ),
        ],
    )
    @pytest.mark.parametrize("inplace", [True, False], ids=["inplace", "not_inplace"])
    @backend_test_context(
        test_backend="Big_Data_Cloud",
        choices=("Big_Data_Cloud", "Small_Data_Local"),
    )
    def test_pinned_groupby_prevents_backend_switch(
        self, groupby_class, groupby_operation, agg_operation, inplace
    ):
        """Test that pinning a GroupBy object prevents operations from switching backends."""
        modin_df, pandas_df = create_test_dfs(
            {
                "col0": list(range(BIG_DATA_CLOUD_MIN_NUM_ROWS - 1)),
                "col1": list(range(1, BIG_DATA_CLOUD_MIN_NUM_ROWS)),
            }
        )

        assert modin_df.get_backend() == "Big_Data_Cloud"

        # Create groupby object and pin it
        modin_groupby = groupby_operation(modin_df)
        pandas_groupby = groupby_operation(pandas_df)

        if inplace:
            modin_groupby.pin_backend(inplace=True)
            assert modin_groupby.is_backend_pinned()
        else:
            pinned_groupby = modin_groupby.pin_backend(inplace=False)
            assert not modin_groupby.is_backend_pinned()
            assert pinned_groupby.is_backend_pinned()
            modin_groupby = pinned_groupby

        # Register a post-op switch that would normally move to Small_Data_Local
        register_function_for_post_op_switch(
            class_name=groupby_class, backend="Big_Data_Cloud", method="sum"
        )

        # The operation should stay on Big_Data_Cloud due to pinning
        modin_result = agg_operation(modin_groupby)
        pandas_result = agg_operation(pandas_groupby)
        df_equals(modin_result, pandas_result)
        assert modin_result.get_backend() == "Big_Data_Cloud"


class TestSwitchBackendPreOp:
    @pytest.mark.parametrize(
        "data_size, expected_backend",
        [
            param(
                BIG_DATA_CLOUD_MIN_NUM_ROWS - 1,
                "Small_Data_Local",
                id="small_data_should_move_to_small_engine",
            ),
            param(
                BIG_DATA_CLOUD_MIN_NUM_ROWS,
                "Big_Data_Cloud",
                id="big_data_should_stay_in_cloud",
            ),
        ],
    )
    def test_describe_switches_depending_on_data_size(
        self, data_size, expected_backend
    ):
        # Mock the default describe() implementation so that we can check that we
        # are calling it with the correct backend as an input. We can't just inspect
        # the mock's call_args_list because call_args_list keeps a reference to the
        # input dataframe, whose backend may change in place.
        mock_describe = mock.Mock(
            wraps=pd.DataFrame._extensions[None]["describe"],
            side_effect=(
                # 1) Record the input backend
                lambda self, *args, **kwargs: setattr(
                    mock_describe, "_last_input_backend", self.get_backend()
                )
                # 2) Return mock.DEFAULT so that we fall back to the original
                #    describe() implementation
                or mock.DEFAULT
            ),
        )
        with backend_test_context(
            test_backend="Big_Data_Cloud",
            choices=("Big_Data_Cloud", "Small_Data_Local"),
        ):
            df = pd.DataFrame(list(range(data_size)))
            with mock.patch.dict(
                pd.DataFrame._extensions[None], {"describe": mock_describe}
            ):
                # Before we register the post-op switch, the describe() method
                # should not trigger auto-switch.
                assert df.get_backend() == "Big_Data_Cloud"
                describe_result = df.describe()
                df_equals(describe_result, df._to_pandas().describe())
                assert describe_result.get_backend() == "Big_Data_Cloud"
                assert df.get_backend() == "Big_Data_Cloud"
                mock_describe.assert_called_once()
                assert mock_describe._last_input_backend == "Big_Data_Cloud"

                mock_describe.reset_mock()

                register_function_for_pre_op_switch(
                    class_name="DataFrame", backend="Big_Data_Cloud", method="describe"
                )

                # Now that we've registered the pre-op switch, the describe() call
                # should trigger auto-switch.
                assert df.get_backend() == "Big_Data_Cloud"
                describe_result = df.describe()
                df_equals(describe_result, df._to_pandas().describe())
                assert describe_result.get_backend() == expected_backend
                assert df.get_backend() == expected_backend
                mock_describe.assert_called_once()
                assert mock_describe._last_input_backend == expected_backend

    def test_read_json_with_extensions(self):
        json_input = json.dumps({"col0": [1]})
        # Mock the read_json implementation for each backend so that we can check
        # that we are calling the correct implementation. Also, we have to make
        # the extension methods produce dataframes with the correct backends.
        pandas_read_json = mock.Mock(
            wraps=(
                lambda *args, **kwargs: _GENERAL_EXTENSIONS[None]["read_json"](
                    *args, **kwargs
                ).move_to("Small_Data_Local")
            )
        )
        pandas_read_json.__name__ = "read_json"
        cloud_read_json = mock.Mock(
            wraps=(
                lambda *args, **kwargs: _GENERAL_EXTENSIONS[None]["read_json"](
                    *args, **kwargs
                ).move_to("Big_Data_Cloud")
            )
        )
        cloud_read_json.__name__ = "read_json"

        register_pd_accessor("read_json", backend="Small_Data_Local")(pandas_read_json)
        register_pd_accessor("read_json", backend="Big_Data_Cloud")(cloud_read_json)

        with backend_test_context(
            test_backend="Big_Data_Cloud",
            choices=("Big_Data_Cloud", "Small_Data_Local"),
        ):
            df = pd.read_json(StringIO(json_input))
            assert df.get_backend() == "Big_Data_Cloud"
            pandas_read_json.assert_not_called()
            cloud_read_json.assert_called_once()

            register_function_for_pre_op_switch(
                class_name=None, backend="Big_Data_Cloud", method="read_json"
            )

            pandas_read_json.reset_mock()
            cloud_read_json.reset_mock()

            df = pd.read_json(StringIO(json_input))

            assert df.get_backend() == "Small_Data_Local"
            pandas_read_json.assert_called_once()
            cloud_read_json.assert_not_called()

    def test_read_json_without_extensions(self):
        json_input = json.dumps({"col0": [1]})

        with backend_test_context(
            test_backend="Big_Data_Cloud",
            choices=("Big_Data_Cloud", "Small_Data_Local"),
        ):
            df = pd.read_json(StringIO(json_input))
            assert df.get_backend() == "Big_Data_Cloud"

            register_function_for_pre_op_switch(
                class_name=None, backend="Big_Data_Cloud", method="read_json"
            )

            df = pd.read_json(StringIO(json_input))

            assert df.get_backend() == "Small_Data_Local"

    @pytest.mark.parametrize(
        "data_size, expected_backend",
        [
            param(
                BIG_DATA_CLOUD_MIN_NUM_ROWS - 1,
                "Small_Data_Local",
                id="small_data_should_move_to_small_engine",
            ),
            param(
                BIG_DATA_CLOUD_MIN_NUM_ROWS,
                "Big_Data_Cloud",
                id="big_data_should_stay_in_cloud",
            ),
        ],
    )
    def test_iloc_setitem_switches_depending_on_data_size(
        self, data_size, expected_backend
    ):
        with backend_test_context(
            test_backend="Big_Data_Cloud",
            choices=("Big_Data_Cloud", "Small_Data_Local"),
        ):
            md_df, pd_df = create_test_dfs(list(range(data_size)))
            assert md_df.get_backend() == "Big_Data_Cloud"
            eval_general(
                md_df,
                pd_df,
                lambda df: df.iloc.__setitem__((0, 0), -1),
                __inplace__=True,
            )
            assert md_df.get_backend() == "Big_Data_Cloud"

            register_function_for_pre_op_switch(
                class_name="_iLocIndexer",
                backend="Big_Data_Cloud",
                method="__setitem__",
            )
            eval_general(
                md_df,
                pd_df,
                lambda df: df.iloc.__setitem__((0, 0), 0),
                __inplace__=True,
            )
            assert md_df.get_backend() == expected_backend

    def test_iloc_pinned(self):
        # The operation in test_iloc would naturally cause an automatic switch, but the
        # absence of AutoSwitchBackend or the presence of a pin on the frame prevent this
        # switch from happening.
        data_size = BIG_DATA_CLOUD_MIN_NUM_ROWS - 1
        with backend_test_context(
            test_backend="Big_Data_Cloud",
            choices=("Big_Data_Cloud", "Small_Data_Local"),
        ):
            register_function_for_pre_op_switch(
                class_name="_iLocIndexer",
                backend="Big_Data_Cloud",
                method="__setitem__",
            )
            # No pin or config, should switch
            df = pd.DataFrame(list(range(data_size)))
            assert df.get_backend() == "Big_Data_Cloud"
            df.iloc[(0, 0)] = -1
            assert df.get_backend() == "Small_Data_Local"
            # config set to false, should not switch
            with config_context(AutoSwitchBackend=False):
                df = pd.DataFrame(list(range(data_size)))
                assert df.get_backend() == "Big_Data_Cloud"
                df.iloc[(0, 0)] = -2
                assert df.get_backend() == "Big_Data_Cloud"
            # no config, but data is pinned
            df = pd.DataFrame(list(range(data_size))).pin_backend()
            assert df.get_backend() == "Big_Data_Cloud"
            df.iloc[(0, 0)] = -3
            assert df.get_backend() == "Big_Data_Cloud"
            # a frame-level pin remains valid across a transformation
            df_copy = df + 1
            assert df_copy.get_backend() == "Big_Data_Cloud"
            df_copy.iloc[(0, 0)] = -4
            assert df_copy.get_backend() == "Big_Data_Cloud"
            # unpinning df allows a switch again
            df.unpin_backend(inplace=True)
            assert df.get_backend() == "Big_Data_Cloud"
            df.iloc[(0, 0)] = -5
            assert df.get_backend() == "Small_Data_Local"
            # An in-place set_backend operation clears the pin
            df.move_to("Big_Data_Cloud", inplace=True)
            # check in-place pin/unpin operations
            df.pin_backend(inplace=True)
            assert df.get_backend() == "Big_Data_Cloud"
            df.iloc[(0, 0)] = -6
            assert df.get_backend() == "Big_Data_Cloud"
            df.unpin_backend(inplace=True)
            assert df.get_backend() == "Big_Data_Cloud"
            df.iloc[(0, 0)] = -7
            assert df.get_backend() == "Small_Data_Local"

    @pytest.mark.parametrize(
        "args, kwargs, expected_backend",
        (
            param((), {}, "Small_Data_Local", id="no_args_or_kwargs"),
            param(([1],), {}, "Small_Data_Local", id="small_list_data_in_arg"),
            param(
                (list(range(BIG_DATA_CLOUD_MIN_NUM_ROWS)),),
                {},
                "Small_Data_Local",
                id="big_list_data_in_arg",
            ),
            param((), {"data": [1]}, "Small_Data_Local", id="list_data_in_kwarg"),
            param(
                (),
                {"data": pandas.Series([1])},
                "Small_Data_Local",
                id="series_data_in_kwarg",
            ),
            param(
                (),
                {"query_compiler": CloudForBigDataQC(pandas.DataFrame([0, 1, 2]))},
                "Big_Data_Cloud",
                id="cloud_query_compiler_in_kwarg",
            ),
            param(
                (),
                {"query_compiler": LocalForSmallDataQC(pandas.DataFrame([0, 1, 2]))},
                "Small_Data_Local",
                id="small_query_compiler_in_kwarg",
            ),
        ),
    )
    @pytest.mark.parametrize("data_class", [pd.DataFrame, pd.Series])
    def test___init___with_in_memory_data_uses_native_query_compiler(
        self, args, kwargs, expected_backend, data_class
    ):
        register_function_for_pre_op_switch(
            class_name=data_class.__name__,
            method="__init__",
            backend="Big_Data_Cloud",
        )
        with backend_test_context(
            test_backend="Big_Data_Cloud",
            choices=("Big_Data_Cloud", "Small_Data_Local"),
        ):
            assert data_class(*args, **kwargs).get_backend() == expected_backend

    @pytest.mark.parametrize(
        "num_input_rows, expected_backend",
        [
            param(
                BIG_DATA_CLOUD_MIN_NUM_ROWS - 1,
                "Small_Data_Local",
            ),
            (BIG_DATA_CLOUD_MIN_NUM_ROWS, "Big_Data_Cloud"),
        ],
    )
    @pytest.mark.parametrize(
        "groupby_class,operation",
        [
            param(
                "DataFrameGroupBy",
                lambda df: df.groupby("col0").apply(lambda x: x + 1),
                id="DataFrameGroupBy",
            ),
            param(
                "SeriesGroupBy",
                lambda df: df.groupby("col0")["col1"].apply(lambda x: x + 1),
                id="SeriesGroupBy",
            ),
        ],
    )
    def test_groupby_apply_switches_for_small_input(
        self, num_input_rows, expected_backend, operation, groupby_class
    ):
        with backend_test_context(
            test_backend="Big_Data_Cloud",
            choices=("Big_Data_Cloud", "Small_Data_Local"),
        ):
            modin_df, pandas_df = create_test_dfs(
                {
                    "col0": list(range(num_input_rows)),
                    "col1": list(range(1, num_input_rows + 1)),
                }
            )
            assert modin_df.get_backend() == "Big_Data_Cloud"
            assert operation(modin_df).get_backend() == "Big_Data_Cloud"

            register_function_for_pre_op_switch(
                class_name=groupby_class, backend="Big_Data_Cloud", method="apply"
            )

            modin_result = operation(modin_df)
            pandas_result = operation(pandas_df)
            df_equals(modin_result, pandas_result)
            assert modin_result.get_backend() == expected_backend
            assert modin_df.get_backend() == expected_backend


def test_move_to_clears_pin():
    # Pin status is reset to false after a set_backend call
    with backend_test_context(
        test_backend="Big_Data_Cloud",
        choices=("Big_Data_Cloud", "Small_Data_Local"),
    ):
        df = pd.DataFrame(list(range(10)))
        # in-place
        df.pin_backend(inplace=True)
        assert df.is_backend_pinned()
        df.move_to("Small_Data_Local", inplace=True)
        assert not df.is_backend_pinned()
        # not in-place
        intermediate = df.pin_backend().move_to("Big_Data_Cloud")
        assert not intermediate.is_backend_pinned()
        assert intermediate.pin_backend().is_backend_pinned()


@pytest.mark.parametrize(
    "pin_backends, expected_backend",
    [
        param(
            [("Small_Data_Local", False), ("Big_Data_Cloud", False)],
            "Small_Data_Local",
            id="no_pin",
        ),  # no backend pinned
        param(
            [("Small_Data_Local", True), ("Big_Data_Cloud", False)],
            "Small_Data_Local",
            id="one_pin",
        ),  # one backend is pinned, so move there
        param(
            [
                ("Big_Data_Cloud", False),
                ("Small_Data_Local", True),
                ("Small_Data_Local", True),
            ],
            "Small_Data_Local",
            id="two_pin",
        ),  # two identical pinned backends
        param(
            [("Small_Data_Local", True), ("Big_Data_Cloud", True)],
            None,
            id="conflict_pin",
        ),  # conflicting pins raises ValueError
    ],
)
def test_concat_with_pin(pin_backends, expected_backend):
    with backend_test_context(
        test_backend="Big_Data_Cloud",
        choices=("Big_Data_Cloud", "Small_Data_Local"),
    ):
        dfs = [
            pd.DataFrame([1] * 10).move_to(backend)._set_backend_pinned(should_pin)
            for backend, should_pin in pin_backends
        ]
        if expected_backend is None:
            with pytest.raises(
                ValueError,
                match="Cannot combine arguments that are pinned to conflicting backends",
            ):
                pd.concat(dfs)
        else:
            result = pd.concat(dfs)
            assert result.is_backend_pinned() == any(
                df.is_backend_pinned() for df in dfs
            )
            assert result.get_backend() == expected_backend
            df_equals(
                result, pandas.concat([pandas.DataFrame([1] * 10)] * len(pin_backends))
            )


@pytest.mark.parametrize(
    "groupby_operation",
    [
        param(
            lambda df: df.groupby("col0"),
            id="DataFrameGroupBy",
        ),
        param(
            lambda df: df.groupby("col0")["col1"],
            id="SeriesGroupBy",
        ),
    ],
)
def test_pin_groupby_in_place(groupby_operation):
    """Test that groupby objects can be pinned with inplace=True."""
    modin_df = pd.DataFrame(
        {
            "col0": list(range(BIG_DATA_CLOUD_MIN_NUM_ROWS - 1)),
            "col1": list(range(1, BIG_DATA_CLOUD_MIN_NUM_ROWS)),
        }
    )

    groupby_object = groupby_operation(modin_df)
    assert not groupby_object.is_backend_pinned()

    groupby_object.pin_backend(inplace=True)
    assert groupby_object.is_backend_pinned()

    groupby_object.unpin_backend(inplace=True)
    assert not groupby_object.is_backend_pinned()


@pytest.mark.parametrize(
    "groupby_operation",
    [
        param(
            lambda df: df.groupby("col0"),
            id="DataFrameGroupBy",
        ),
        param(
            lambda df: df.groupby("col0")["col1"],
            id="SeriesGroupBy",
        ),
    ],
)
def test_pin_groupby_not_in_place(groupby_operation):
    """Test that pin_backend works with inplace=False for groupby objects."""
    original_groupby = groupby_operation(pd.DataFrame(columns=["col0", "col1"]))
    assert not original_groupby.is_backend_pinned()
    new_groupby = original_groupby.pin_backend(inplace=False)
    assert not original_groupby.is_backend_pinned()
    assert new_groupby.is_backend_pinned()


@pytest.mark.parametrize(
    "groupby_operation",
    [
        param(
            lambda df: df.groupby("col0"),
            id="DataFrameGroupBy",
        ),
        param(
            lambda df: df.groupby("col0")["col1"],
            id="SeriesGroupBy",
        ),
    ],
)
def test_unpin_groupby_not_in_place(groupby_operation):
    """Test that unpin_backend works with inplace=False for groupby objects."""
    original_groupby = groupby_operation(pd.DataFrame(columns=["col0", "col1"]))
    original_groupby.pin_backend(inplace=True)
    assert original_groupby.is_backend_pinned()
    new_groupby = original_groupby.unpin_backend(inplace=False)
    assert original_groupby.is_backend_pinned()
    assert not new_groupby.is_backend_pinned()


@pytest.mark.parametrize(
    "data_type,data_factory,groupby_factory",
    [
        param(
            "DataFrame",
            lambda: pd.DataFrame(
                {
                    "col0": list(range(BIG_DATA_CLOUD_MIN_NUM_ROWS - 1)),
                    "col1": list(range(1, BIG_DATA_CLOUD_MIN_NUM_ROWS)),
                }
            ),
            lambda obj: obj.groupby("col0"),
            id="DataFrame",
        ),
        param(
            "Series",
            lambda: pd.Series(list(range(1, BIG_DATA_CLOUD_MIN_NUM_ROWS)), name="data"),
            lambda obj: obj.groupby([0] * (BIG_DATA_CLOUD_MIN_NUM_ROWS - 1)),
            id="Series",
        ),
    ],
)
def test_groupby_pinning_reflects_parent_object_pin_status(
    data_type, data_factory, groupby_factory
):
    """Test that groupby pinning inherits from parent object (DataFrame/Series) pin status but can be modified independently."""
    modin_obj = data_factory()

    old_groupby_obj = groupby_factory(modin_obj)

    # Initially not pinned
    assert not old_groupby_obj.is_backend_pinned()
    assert not modin_obj.is_backend_pinned()

    # Pin the parent object - new groupby objects should inherit this
    modin_obj.pin_backend(inplace=True)

    # Create a new groupby object after pinning parent object
    new_groupby_obj = groupby_factory(modin_obj)

    # New groupby should inherit the pinned status
    assert new_groupby_obj.is_backend_pinned()
    assert modin_obj.is_backend_pinned()

    # But we can still modify groupby pinning independently
    new_groupby_obj.unpin_backend(inplace=True)

    # Parent object should remain pinned, groupby should be unpinned
    assert not new_groupby_obj.is_backend_pinned()
    assert modin_obj.is_backend_pinned()

    assert not old_groupby_obj.is_backend_pinned()
    old_groupby_obj.pin_backend(inplace=True)
    assert old_groupby_obj.is_backend_pinned()


def test_second_init_only_calls_from_pandas_once_github_issue_7559():
    with config_context(Backend="Big_Data_Cloud"):
        # Create a dataframe once first so that we can initialize the dummy
        # query compiler for the Big_Data_Cloud backend.
        pd.DataFrame([1])
        with mock.patch.object(
            factories.Big_Data_CloudOnNativeFactory.io_cls.query_compiler_cls,
            "from_pandas",
            wraps=factories.Big_Data_CloudOnNativeFactory.io_cls.query_compiler_cls.from_pandas,
        ) as mock_from_pandas:
            pd.DataFrame([1])
            mock_from_pandas.assert_called_once()


def test_native_config():
    qc = NativeQueryCompiler(pandas.DataFrame([0, 1, 2]))

    # Native Query Compiler gets a special configuration
    assert qc._TRANSFER_THRESHOLD == 0
    assert qc._transfer_threshold() == NativePandasTransferThreshold.get()
    assert qc._MAX_SIZE_THIS_ENGINE_CAN_HANDLE == 1
    assert qc._engine_max_size() == NativePandasMaxRows.get()

    oldmax = qc._engine_max_size()
    oldthresh = qc._transfer_threshold()

    with config_context(NativePandasMaxRows=123, NativePandasTransferThreshold=321):
        qc2 = NativeQueryCompiler(pandas.DataFrame([0, 1, 2]))
        assert qc2._transfer_threshold() == 321
        assert qc2._engine_max_size() == 123
        assert qc._engine_max_size() == 123
        assert qc._transfer_threshold() == 321

        # sub class configuration is unchanged
        class AQC(NativeQueryCompiler):
            pass

        subqc = AQC(pandas.DataFrame([0, 1, 2]))
        assert subqc._TRANSFER_THRESHOLD == 0
        assert subqc._MAX_SIZE_THIS_ENGINE_CAN_HANDLE == 1

    assert qc._engine_max_size() == oldmax
    assert qc._transfer_threshold() == oldthresh


def test_cast_metrics(pico_df, cluster_df):
    try:
        count = 0

        def test_handler(metric: str, value) -> None:
            nonlocal count
            if metric.startswith("modin.hybrid.merge"):
                count += 1

        add_metric_handler(test_handler)
        df3 = pd.concat([pico_df, cluster_df], axis=1)
        assert df3.get_backend() == "Cluster"  # result should be on cluster
        assert count == 7
    finally:
        clear_metric_handler(test_handler)


def test_switch_metrics(pico_df, cluster_df):
    with backend_test_context(
        test_backend="Big_Data_Cloud",
        choices=("Big_Data_Cloud", "Small_Data_Local"),
    ):
        try:
            count = 0

            def test_handler(metric: str, value) -> None:
                nonlocal count
                if metric.startswith("modin.hybrid.auto"):
                    count += 1

            add_metric_handler(test_handler)

            register_function_for_pre_op_switch(
                class_name="DataFrame",
                backend="Big_Data_Cloud",
                method="describe",
            )
            df = pd.DataFrame([1] * 10)
            assert df.get_backend() == "Big_Data_Cloud"
            df.describe()
            assert count == 8
        finally:
            clear_metric_handler(test_handler)
