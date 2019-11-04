import pandas

from modin.backends.pandas.query_compiler import PandasQueryCompiler
from modin.engines.ray.generic.io import RayIO
from modin.engines.base.io import CSVReader, JSONReader
from modin.backends.pandas.parsers import (
    PandasCSVParser,
    PandasJSONParser,
    _split_result_for_readers,
)
from modin.engines.ray.task_wrapper import RayTask
from modin.engines.ray.pandas_on_ray.frame.partition import PandasOnRayFramePartition
from modin.engines.ray.pandas_on_ray.frame.data import PandasOnRayFrame
from modin import __execution_engine__

if __execution_engine__ == "Ray":
    import ray

    @ray.remote
    def _read_parquet_columns(path, columns, num_splits, kwargs):  # pragma: no cover
        """Use a Ray task to read columns from Parquet into a Pandas DataFrame.

        Note: Ray functions are not detected by codecov (thus pragma: no cover)

        Args:
            path: The path of the Parquet file.
            columns: The list of column names to read.
            num_splits: The number of partitions to split the column into.

        Returns:
             A list containing the split Pandas DataFrames and the Index as the last
                element. If there is not `index_col` set, then we just return the length.
                This is used to determine the total length of the DataFrame to build a
                default Index.
        """
        import pyarrow.parquet as pq

        kwargs["use_pandas_metadata"] = True
        df = pq.read_table(path, columns=columns, **kwargs).to_pandas()
        df = df[columns]
        # Append the length of the index here to build it externally
        return _split_result_for_readers(0, num_splits, df) + [len(df.index), df.dtypes]

    @ray.remote
    def _read_hdf_columns(path_or_buf, columns, num_splits, kwargs):  # pragma: no cover
        """Use a Ray task to read columns from HDF5 into a Pandas DataFrame.

        Note: Ray functions are not detected by codecov (thus pragma: no cover)

        Args:
            path_or_buf: The path of the HDF5 file.
            columns: The list of column names to read.
            num_splits: The number of partitions to split the column into.

        Returns:
             A list containing the split Pandas DataFrames and the Index as the last
                element. If there is not `index_col` set, then we just return the length.
                This is used to determine the total length of the DataFrame to build a
                default Index.
        """

        df = pandas.read_hdf(path_or_buf, columns=columns, **kwargs)
        # Append the length of the index here to build it externally
        return _split_result_for_readers(0, num_splits, df) + [len(df.index)]

    @ray.remote
    def _read_feather_columns(path, columns, num_splits):  # pragma: no cover
        """Use a Ray task to read columns from Feather into a Pandas DataFrame.

        Note: Ray functions are not detected by codecov (thus pragma: no cover)

        Args:
            path: The path of the Feather file.
            columns: The list of column names to read.
            num_splits: The number of partitions to split the column into.

        Returns:
             A list containing the split Pandas DataFrames and the Index as the last
                element. If there is not `index_col` set, then we just return the length.
                This is used to determine the total length of the DataFrame to build a
                default Index.
        """
        from pyarrow import feather

        df = feather.read_feather(path, columns=columns)
        # Append the length of the index here to build it externally
        return _split_result_for_readers(0, num_splits, df) + [len(df.index)]

    @ray.remote
    def _read_sql_with_limit_offset(
        num_splits, sql, con, index_col, kwargs
    ):  # pragma: no cover
        """Use a Ray task to read a chunk of SQL source.

        Note: Ray functions are not detected by codecov (thus pragma: no cover)
        """
        pandas_df = pandas.read_sql(sql, con, index_col=index_col, **kwargs)
        if index_col is None:
            index = len(pandas_df)
        else:
            index = pandas_df.index
        return _split_result_for_readers(1, num_splits, pandas_df) + [index]


class PandasOnRayCSVReader(RayTask, PandasCSVParser, CSVReader):
    frame_cls = PandasOnRayFrame
    frame_partition_cls = PandasOnRayFramePartition
    query_compiler_cls = PandasQueryCompiler


class PandasOnRayJSONReader(RayTask, PandasJSONParser, JSONReader):
    frame_cls = PandasOnRayFrame
    frame_partition_cls = PandasOnRayFramePartition
    query_compiler_cls = PandasQueryCompiler


class PandasOnRayIO(RayIO):

    frame_partition_cls = PandasOnRayFramePartition
    query_compiler_cls = PandasQueryCompiler
    frame_cls = PandasOnRayFrame

    read_parquet_remote_task = _read_parquet_columns
    read_hdf_remote_task = _read_hdf_columns
    read_feather_remote_task = _read_feather_columns
    read_sql_remote_task = _read_sql_with_limit_offset

    read_csv = PandasOnRayCSVReader.read
    read_json = PandasOnRayJSONReader.read
