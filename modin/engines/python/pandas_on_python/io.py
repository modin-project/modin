from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from modin.engines.base.io import BaseIO
from modin.data_management.query_compiler import PandasQueryCompiler
from modin.engines.python.pandas_on_python.frame.partition_manager import (
    PythonFramePartitionManager,
)


class PandasOnPythonIO(BaseIO):

    block_partitions_cls = PythonFramePartitionManager
    query_compiler_cls = PandasQueryCompiler
