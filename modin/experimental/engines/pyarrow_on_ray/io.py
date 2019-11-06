from modin.backends.pyarrow.query_compiler import PyarrowQueryCompiler
from modin.engines.ray.generic.io import RayIO
from modin.experimental.engines.pyarrow_on_ray.frame.data import PyarrowOnRayFrame
from modin.experimental.engines.pyarrow_on_ray.frame.partition import (
    PyarrowOnRayFramePartition,
)
from modin.backends.pyarrow.parsers import PyarrowCSVParser
from modin.engines.ray.task_wrapper import RayTask
from modin.engines.base.io import CSVReader


class PyarrowOnRayCSVReader(RayTask, PyarrowCSVParser, CSVReader):
    frame_cls = PyarrowOnRayFrame
    frame_partition_cls = PyarrowOnRayFramePartition
    query_compiler_cls = PyarrowQueryCompiler


class PyarrowOnRayIO(RayIO):
    frame_cls = PyarrowOnRayFrame
    frame_partition_cls = PyarrowOnRayFramePartition
    query_compiler_cls = PyarrowQueryCompiler
    csv_reader = PyarrowOnRayCSVReader

    read_parquet_remote_task = None
    read_hdf_remote_task = None
    read_feather_remote_task = None
    read_sql_remote_task = None
