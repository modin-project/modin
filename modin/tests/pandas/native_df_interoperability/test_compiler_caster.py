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

import json
from io import StringIO
from unittest import mock

import pandas
import pytest
from pytest import param

import modin.pandas as pd
from modin.config import context as config_context
from modin.config.envvars import Backend, Engine, Execution
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
from modin.pandas.api.extensions import register_pd_accessor
from modin.tests.pandas.utils import create_test_dfs, df_equals, eval_general

BIG_DATA_CLOUD_MIN_NUM_ROWS = 10


class CloudQC(NativeQueryCompiler):
    "Represents a cloud-hosted query compiler"

    def get_backend(self):
        return "Cloud"

    def max_cost(self):
        return QCCoercionCost.COST_IMPOSSIBLE

    def move_to_cost(self, other_qc_cls, api_cls_name, op):
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
            ClusterQC: QCCoercionCost.COST_MEDIUM,
            DefaultQC: QCCoercionCost.COST_MEDIUM,
            LocalMachineQC: QCCoercionCost.COST_HIGH,
            PicoQC: QCCoercionCost.COST_IMPOSSIBLE,
            OmniscientEagerQC: None,
            OmniscientLazyQC: None,
        }[other_qc_cls]

    def stay_cost(self, api_cls_name, op):
        return QCCoercionCost.COST_HIGH


class ClusterQC(NativeQueryCompiler):
    "Represents a local network cluster query compiler"

    def get_backend(self):
        return "Cluster"

    def max_cost(self):
        return QCCoercionCost.COST_HIGH

    def move_to_cost(self, other_qc_cls, api_cls_name, op):
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

    def max_cost(self):
        return QCCoercionCost.COST_MEDIUM

    def move_to_cost(self, other_qc_cls, api_cls_name, op):
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

    def max_cost(self):
        return QCCoercionCost.COST_LOW

    def move_to_cost(self, other_qc_cls, api_cls_name, op):
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

    def move_to_cost(self, other_qc_cls, api_cls_name, op):
        return {
            CloudQC: -1000,
            ClusterQC: 10000,
            AdversarialQC: QCCoercionCost.COST_ZERO,
        }[other_qc_cls]


class OmniscientEagerQC(NativeQueryCompiler):
    "Represents a query compiler which knows a lot, and wants to steal work"

    def get_backend(self):
        return "Eager"

    # keep other workloads from getting my workload
    def move_to_cost(self, other_qc_cls, api_cls_name, op):
        if OmniscientEagerQC is other_qc_cls:
            return QCCoercionCost.COST_ZERO
        return QCCoercionCost.COST_IMPOSSIBLE

    # try to force other workloads to my engine
    @classmethod
    def move_to_me_cost(cls, other_qc, api_cls_name, op):
        return QCCoercionCost.COST_ZERO


class OmniscientLazyQC(NativeQueryCompiler):
    "Represents a query compiler which knows a lot, and wants to avoid work"

    def get_backend(self):
        return "Lazy"

    # encorage other engines to take my workload
    def move_to_cost(self, other_qc_cls, api_cls_name, op):
        return QCCoercionCost.COST_ZERO

    # try to keep other workloads from getting my workload
    @classmethod
    def move_to_me_cost(cls, other_qc, api_cls_name, op):
        if isinstance(other_qc, cls):
            return QCCoercionCost.COST_ZERO
        return QCCoercionCost.COST_IMPOSSIBLE


class DefaultQC(NativeQueryCompiler):
    "Represents a query compiler with no costing information"

    def get_backend(self):
        return "Test_casting_default"


class DefaultQC2(NativeQueryCompiler):
    "Represents a query compiler with no costing information, but different."

    def get_backend(self):
        return "Test_casting_default_2"


class CloudForBigDataQC(NativeQueryCompiler):
    """Represents a cloud-hosted query compiler that prefers to stay on the cloud only for big data"""

    def get_backend(self) -> str:
        return "Big_Data_Cloud"

    def move_to_cost(self, other_qc_type, api_cls_name, operation):
        return (
            QCCoercionCost.COST_LOW
            if other_qc_type is NativeQueryCompiler
            and self.get_axis_len(axis=0) < BIG_DATA_CLOUD_MIN_NUM_ROWS
            else None
        )

    def stay_cost(self, api_cls_name, operation):
        return QCCoercionCost.COST_MEDIUM


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
register_backend("Local_Machine", LocalMachineQC)
register_backend("Adversarial", AdversarialQC)
register_backend("Eager", OmniscientEagerQC)
register_backend("Lazy", OmniscientLazyQC)
register_backend("Test_Casting_Default", DefaultQC)
register_backend("Test_Casting_Default_2", DefaultQC2)
register_backend("Big_Data_Cloud", CloudForBigDataQC)


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


def test_cast_to_second_backend_with_concat(pico_df, cluster_df):
    df3 = pd.concat([pico_df, cluster_df], axis=1)
    assert pico_df.get_backend() == "Pico"
    assert cluster_df.get_backend() == "Cluster"
    assert df3.get_backend() == "Cluster"  # result should be on cluster


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
        default_df._query_compiler.move_to_cost(type(default2_df._query_compiler))
        is None
    )
    assert (
        default_df._query_compiler.move_to_cost(type(default_df._query_compiler))
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


def test_switch_local_to_cloud_with_iloc___setitem__(local_df, cloud_df):
    local_df.iloc[:, 0] = cloud_df.iloc[:, 0] + 1
    expected_pandas = local_df._to_pandas()
    expected_pandas.iloc[:, 0] = cloud_df._to_pandas().iloc[:, 0] + 1
    df_equals(local_df, expected_pandas)
    assert local_df.get_backend() == "Cloud"


# Outlines a future generic function for determining when to stay
# or move to different engines. In the current state it is pretty
# trivial, but added for completeness
def test_stay_or_move_evaluation(cloud_df, default_df):
    default_cls = type(default_df._get_query_compiler())
    cloud_cls = type(cloud_df._get_query_compiler())

    stay_cost = cloud_df._get_query_compiler().stay_cost("Series", "myop")
    move_cost = cloud_df._get_query_compiler().move_to_cost(
        default_cls, "Series", "myop"
    )
    df = cloud_df
    if stay_cost > move_cost:
        df = cloud_df.move_to("Test_casting_default")
    else:
        assert False

    stay_cost = df._get_query_compiler().stay_cost("Series", "myop")
    move_cost = df._get_query_compiler().move_to_cost(cloud_cls, "Series", "myop")
    assert stay_cost is None
    assert move_cost is None


class TestSwitchBackendPostOpDependingOnDataSize:
    def test_read_json(self):
        with config_context(Backend="Big_Data_Cloud"):
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
            assert pd.read_json(StringIO(small_json)).get_backend() == "Pandas"

    def test_agg(self):
        with config_context(Backend="Big_Data_Cloud"):
            df = pd.DataFrame([[1, 2], [3, 4]])
            assert df.get_backend() == "Big_Data_Cloud"
            assert df.sum().get_backend() == "Big_Data_Cloud"
            register_function_for_post_op_switch(
                class_name="DataFrame", backend="Big_Data_Cloud", method="sum"
            )
            assert df.get_backend() == "Big_Data_Cloud"
            assert df.sum().get_backend() == "Pandas"


class TestSwitchBackendPreOp:
    @pytest.mark.parametrize(
        "data_size, expected_backend",
        [
            param(
                BIG_DATA_CLOUD_MIN_NUM_ROWS - 1,
                "Pandas",
                id="small_data_should_move_to_pandas",
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
        with config_context(Backend="Big_Data_Cloud"):
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
                ).move_to("Pandas")
            )
        )
        cloud_read_json = mock.Mock(
            wraps=(
                lambda *args, **kwargs: _GENERAL_EXTENSIONS[None]["read_json"](
                    *args, **kwargs
                ).move_to("Big_Data_Cloud")
            )
        )

        def move_to_cost(self, other_qc_type, api_cls_name, operation):
            """Make read_json() always move to Pandas backend."""
            return (
                QCCoercionCost.COST_LOW
                if other_qc_type is NativeQueryCompiler and operation == "read_json"
                else None
            )

        register_pd_accessor("read_json", backend="Pandas")(pandas_read_json)
        register_pd_accessor("read_json", backend="Big_Data_Cloud")(cloud_read_json)

        with config_context(Backend="Big_Data_Cloud"), mock.patch.object(
            CloudForBigDataQC, "move_to_cost", move_to_cost
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

            assert df.get_backend() == "Pandas"
            pandas_read_json.assert_called_once()
            cloud_read_json.assert_not_called()

    def test_read_json_without_extensions(self):
        json_input = json.dumps({"col0": [1]})

        def move_to_cost(self, other_qc_type, api_cls_name, operation):
            """Make read_json() always move to Pandas backend."""
            return (
                QCCoercionCost.COST_LOW
                if other_qc_type is NativeQueryCompiler and operation == "read_json"
                else None
            )

        with config_context(Backend="Big_Data_Cloud"), mock.patch.object(
            CloudForBigDataQC, "move_to_cost", move_to_cost
        ):
            df = pd.read_json(StringIO(json_input))
            assert df.get_backend() == "Big_Data_Cloud"

            register_function_for_pre_op_switch(
                class_name=None, backend="Big_Data_Cloud", method="read_json"
            )

            df = pd.read_json(StringIO(json_input))

            assert df.get_backend() == "Pandas"

    @pytest.mark.parametrize(
        "data_size, expected_backend",
        [
            param(
                BIG_DATA_CLOUD_MIN_NUM_ROWS - 1,
                "Pandas",
                id="small_data_should_move_to_pandas",
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
        with config_context(Backend="Big_Data_Cloud"):
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
