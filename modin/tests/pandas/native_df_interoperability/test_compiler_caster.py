

import pandas
from modin.core.storage_formats.base.query_compiler import BaseQueryCompiler, QCCoercionCost
from modin.core.storage_formats.pandas.native_query_compiler import NativeQueryCompiler
from modin.core.storage_formats.pandas.query_compiler import PandasQueryCompiler
from modin.utils import _inherit_docstrings


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

def test_two_same_qc_types_noop():
    df = PicoQC(pandas.DataFrame([0, 1, 2]))
    df2 = PicoQC(pandas.DataFrame([0, 1, 2]))
    df3 = df.concat(axis=1, other=df2)
    assert(type(df3) == type(df2))

def test_two_two_qc_types_rhs():
    df = PicoQC(pandas.DataFrame([0, 1, 2]))
    df2 = ClusterQC(pandas.DataFrame([0, 1, 2]))
    df3 = df.concat(axis=1, other=df2)
    assert(type(df3) == type(df2))

def test_two_two_qc_types_lhs():
    df = PicoQC(pandas.DataFrame([0, 1, 2]))
    df2 = ClusterQC(pandas.DataFrame([0, 1, 2]))
    df3 = df2.concat(axis=1, other=df)
    assert(type(df3) == type(df2)) # should move to cluster

def test_three_two_qc_types_rhs():
    pass

def test_three_two_qc_types_lhs():
    pass

def test_three_two_qc_types_middle():
    pass

def test_three_three_qc_types_rhs():
    pass

def test_three_three_qc_types_lhs():
    pass

def test_three_three_qc_types_middle():
    pass
