from modin.backends.pyarrow.query_compiler import PyarrowQueryCompiler
from modin.engines.ray.generic.io import RayIO
from modin.experimental.engines.pyarrow_on_ray.frame.partition_manager import (
    PyarrowOnRayFrameManager,
)
from modin.experimental.engines.pyarrow_on_ray.frame.partition import (
    PyarrowOnRayFramePartition,
)


class PyarrowOnRayIO(RayIO):

    frame_mgr_cls = PyarrowOnRayFrameManager
    frame_partition_cls = PyarrowOnRayFramePartition
    query_compiler_cls = PyarrowQueryCompiler

    read_parquet_remote_task = None
    read_csv_remote_task = None
    read_hdf_remote_task = None
    read_feather_remote_task = None
    read_sql_remote_task = None
