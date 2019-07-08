from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from modin.engines.base.io import BaseIO
from modin.backends.pandas.query_compiler import PandasQueryCompiler
from modin.engines.multiprocess.pandas_on_multiprocess.frame.partition_manager import (
    MultiprocessFrameManager,
)


class PandasOnMultiprocessIO(BaseIO):

    frame_mgr_cls = MultiprocessFrameManager
    query_compiler_cls = PandasQueryCompiler
