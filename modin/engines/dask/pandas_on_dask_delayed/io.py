from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from modin.engines.base.io import BaseIO
from modin.data_management.query_compiler import PandasQueryCompiler
from modin.engines.dask.pandas_on_dask_delayed.frame.partition_manager import (
    DaskFrameManager,
)


class PandasOnDaskIO(BaseIO):

    block_partitions_cls = DaskFrameManager
    query_compiler_cls = PandasQueryCompiler
