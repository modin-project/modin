from modin.engines.base.io import BaseIO
from modin.backends.pandas.query_compiler import PandasQueryCompiler
from modin.engines.dask.pandas_on_dask_futures.frame.data import PandasOnDaskFrame
from modin.engines.dask.pandas_on_dask_futures.frame.partition import (
    PandasOnDaskFramePartition,
)
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
from modin.engines.dask.task_wrapper import DaskTask


class PandasOnDaskIO(BaseIO):

    frame_cls = PandasOnDaskFrame
    query_compiler_cls = PandasQueryCompiler
    build_args = dict(
        frame_cls=PandasOnDaskFrame,
        frame_partition_cls=PandasOnDaskFramePartition,
        query_compiler_cls=PandasQueryCompiler,
    )

    read_csv = type("", (DaskTask, PandasCSVParser, CSVReader), build_args).read
    read_json = type("", (DaskTask, PandasJSONParser, JSONReader), build_args).read
    read_parquet = type(
        "", (DaskTask, PandasParquetParser, ParquetReader), build_args
    ).read
    # Blocked on pandas-dev/pandas#12236. It is faster to default to pandas.
    # read_hdf = type("", (DaskTask, PandasHDFParser, HDFReader), build_args).read
    read_feather = type(
        "", (DaskTask, PandasFeatherParser, FeatherReader), build_args
    ).read
    read_sql = type("", (DaskTask, PandasSQLParser, SQLReader), build_args).read
