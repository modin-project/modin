from modin.data_management.query_compiler import GandivaQueryCompiler
from modin.engines.base.io import BaseIO
from modin.experimental.engines.gandiva_on_ray.frame.partition_manager import (
    RayFrameManager,
)


class GandivaOnRayIO(BaseIO):

    block_partitions_cls = RayFrameManager
    query_compiler_cls = GandivaQueryCompiler
