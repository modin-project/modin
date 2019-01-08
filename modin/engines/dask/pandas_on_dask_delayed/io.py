from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from modin.engines.base.io import BaseIO
from modin.data_management.query_compiler import PandasQueryCompiler
from .block_partitions import DaskBlockPartitions


class PandasOnDaskIO(BaseIO):

    block_partitions_cls = DaskBlockPartitions
    query_compiler_cls = PandasQueryCompiler
