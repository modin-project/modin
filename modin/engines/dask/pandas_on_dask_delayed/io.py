from modin.engines.base.io import BaseIO
from modin.backends.pandas.query_compiler import PandasQueryCompiler
from modin.engines.dask.pandas_on_dask_delayed.frame.partition_manager import (
    DaskFrameManager,
)


class PandasOnDaskIO(BaseIO):

    frame_mgr_cls = DaskFrameManager
    query_compiler_cls = PandasQueryCompiler
