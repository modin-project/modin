from modin.data_management.query_compiler import GandivaQueryCompiler
from modin.engines.ray.generic.io import RayIO
from modin.experimental.engines.pyarrow_on_ray.frame.partition_manager import (
    PyarrowOnRayFrameManager,
)
from modin.experimental.engines.pyarrow_on_ray.frame.partition import (
    PyarrowOnRayFramePartition,
)


class GandivaOnRayIO(RayIO):

    frame_mgr_cls = PyarrowOnRayFrameManager
    frame_partition_cls = PyarrowOnRayFramePartition
    query_compiler_cls = GandivaQueryCompiler

    read_parquet_remote_task = None
    read_csv_remote_task = None
    read_hdf_remote_task = None
    read_feather_remote_task = None
    read_sql_remote_task = None
