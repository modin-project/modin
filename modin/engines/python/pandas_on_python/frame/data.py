from modin.engines.base.frame.data import BasePandasFrame
from .partition_manager import PythonFrameManager


class PandasOnPythonFrame(BasePandasFrame):

    _frame_mgr_cls = PythonFrameManager
