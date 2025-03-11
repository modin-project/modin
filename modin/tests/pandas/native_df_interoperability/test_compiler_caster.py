

import pandas
import pytest
from modin.core.storage_formats.base.query_compiler import QCCoercionCost
from modin.core.storage_formats.pandas.native_query_compiler import NativeQueryCompiler


class CloudQC(NativeQueryCompiler):
    'Represents a cloud-hosted query compiler'
    def __init__(self, pandas_frame):
        self._modin_frame = pandas_frame
        super().__init__(pandas_frame)
      
    def qc_engine_switch_cost(self, other_qc):
        return {CloudQC: QCCoercionCost.COST_ZERO,
                ClusterQC: QCCoercionCost.COST_MEDIUM,
                LocalMachineQC: QCCoercionCost.COST_HIGH,
                PicoQC: QCCoercionCost.COST_IMPOSSIBLE}

class ClusterQC(NativeQueryCompiler):
    'Represents a local network cluster query compiler'
    def __init__(self, pandas_frame):
        self._modin_frame = pandas_frame
        super().__init__(pandas_frame)
        
    def qc_engine_switch_cost(self, other_qc):
        return {CloudQC: QCCoercionCost.COST_MEDIUM,
                ClusterQC: QCCoercionCost.COST_ZERO,
                LocalMachineQC: QCCoercionCost.COST_MEDIUM,
                PicoQC: QCCoercionCost.COST_HIGH}
    
class LocalMachineQC(NativeQueryCompiler):
    'Represents a local machine query compiler'
    def __init__(self, pandas_frame):
        self._modin_frame = pandas_frame
        super().__init__(pandas_frame)
        
    def qc_engine_switch_cost(self, other_qc):
        return {CloudQC: QCCoercionCost.COST_MEDIUM,
                ClusterQC: QCCoercionCost.COST_LOW,
                LocalMachineQC: QCCoercionCost.COST_ZERO,
                PicoQC: QCCoercionCost.COST_MEDIUM}

class PicoQC(NativeQueryCompiler):
    'Represents a query compiler with very few resources'
    def __init__(self, pandas_frame):
        self._modin_frame = pandas_frame
        super().__init__(pandas_frame)
        
    def qc_engine_switch_cost(self, other_qc):
        return {CloudQC: QCCoercionCost.COST_LOW,
                ClusterQC: QCCoercionCost.COST_LOW,
                LocalMachineQC: QCCoercionCost.COST_LOW,
                PicoQC: QCCoercionCost.COST_ZERO}

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

def test_two_same_qc_types_noop(pico_df):
    df3 = pico_df.concat(axis=1, other=pico_df)
    assert(type(df3) == type(pico_df))

def test_two_two_qc_types_rhs(pico_df, cluster_df):
    df3 = pico_df.concat(axis=1, other=cluster_df)
    assert(type(df3) == type(cluster_df)) # should move to cluster

def test_two_two_qc_types_lhs(pico_df, cluster_df):
    df3 = cluster_df.concat(axis=1, other=pico_df)
    assert(type(df3) == type(cluster_df)) # should move to cluster

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
    assert(type(result) == result_type)

# This currently passes because we have no "max cost" associated
# with a particular QC, so we would move all data to the PicoQC
# As soon as we can represent "max-cost" the result of this operation
# should be to move all dfs to the CloudQC
def test_extreme_pico(pico_df, cloud_df):
    result = cloud_df.concat(axis=1, other=[pico_df, pico_df, pico_df, pico_df, pico_df, pico_df, pico_df])
    assert(type(result) == PicoQC)