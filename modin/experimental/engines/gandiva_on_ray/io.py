from modin.data_management.query_compiler import GandivaQueryCompiler
from modin.engines.base.io import BaseIO
from .block_partitions import RayBlockPartitions


class GandivaOnRayIO(BaseIO):

    block_partitions_cls = RayBlockPartitions
    query_compiler_cls = GandivaQueryCompiler
