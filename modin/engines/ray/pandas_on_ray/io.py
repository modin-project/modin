from modin.backends.pandas.query_compiler import PandasQueryCompiler
from modin.engines.ray.generic.io import RayIO
from modin.engines.base.io import (
    CSVReader,
    JSONReader,
    ParquetReader,
    FeatherReader,
    SQLReader,
)
from modin.backends.pandas.parsers import (
    PandasCSVParser,
    PandasJSONParser,
    PandasParquetParser,
    PandasFeatherParser,
    PandasSQLParser,
)
from modin.engines.ray.task_wrapper import RayTask
from modin.engines.ray.pandas_on_ray.frame.partition import PandasOnRayFramePartition
from modin.engines.ray.pandas_on_ray.frame.data import PandasOnRayFrame


class PandasOnRayIO(RayIO):

    frame_cls = PandasOnRayFrame
    query_compiler_cls = PandasQueryCompiler
    build_args = dict(
        frame_partition_cls=PandasOnRayFramePartition,
        query_compiler_cls=PandasQueryCompiler,
        frame_cls=PandasOnRayFrame,
    )
    read_csv = type("", (RayTask, PandasCSVParser, CSVReader), build_args).read
    read_json = type("", (RayTask, PandasJSONParser, JSONReader), build_args).read
    read_parquet = type(
        "", (RayTask, PandasParquetParser, ParquetReader), build_args
    ).read
    # Blocked on pandas-dev/pandas#12236. It is faster to default to pandas.
    # read_hdf = type("", (RayTask, PandasHDFParser, HDFReader), build_args).read
    read_feather = type(
        "", (RayTask, PandasFeatherParser, FeatherReader), build_args
    ).read
    read_sql = type("", (RayTask, PandasSQLParser, SQLReader), build_args).read
