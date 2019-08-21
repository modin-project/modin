from modin.engines.base.frame.data import BasePandasData
from .partition_manager import PythonFrameManager


class PandasOnPythonData(BasePandasData):

    _frame_mgr_cls = PythonFrameManager
