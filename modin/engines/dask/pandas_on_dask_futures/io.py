from modin.engines.base.io import BaseIO
from modin.backends.pandas.query_compiler import PandasQueryCompiler
from modin.engines.dask.pandas_on_dask_futures.frame.data import PandasOnDaskFrame
from modin.engines.dask.pandas_on_dask_futures.frame.partition import PandasOnDaskFramePartition
from modin.backends.pandas.parsers import (
    PandasCSVParser,
    PandasJSONParser,
)
from modin.engines.dask.task_wrapper import DaskTask
from modin.engines.base.io import CSVReader, JSONReader


class PandasOnDaskCSVReader(DaskTask, PandasCSVParser, CSVReader):
    frame_cls = PandasOnDaskFrame
    frame_partition_cls = PandasOnDaskFramePartition
    query_compiler_cls = PandasQueryCompiler


class PandasOnDaskJSONReader(DaskTask, PandasJSONParser, JSONReader):
    frame_cls = PandasOnDaskFrame
    frame_partition_cls = PandasOnDaskFramePartition
    query_compiler_cls = PandasQueryCompiler


class PandasOnDaskIO(BaseIO):

    frame_cls = PandasOnDaskFrame
    query_compiler_cls = PandasQueryCompiler

    read_csv = PandasOnDaskCSVReader.read
    read_json = PandasOnDaskJSONReader.read
