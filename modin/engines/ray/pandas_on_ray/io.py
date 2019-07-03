from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas

from io import BytesIO
import ray
from modin.data_management.utils import split_result_of_axis_func_pandas
from modin.backends.pandas.query_compiler import PandasQueryCompiler
from modin.engines.ray.generic.io import RayIO, file_open
from modin.engines.ray.pandas_on_ray.frame.partition_manager import (
    PandasOnRayFrameManager,
)
from modin.engines.ray.pandas_on_ray.frame.partition import PandasOnRayFramePartition


def _split_result_for_readers(axis, num_splits, df):  # pragma: no cover
    """Splits the DataFrame read into smaller DataFrames and handles all edge cases.

    Args:
        axis: Which axis to split over.
        num_splits: The number of splits to create.
        df: The DataFrame after it has been read.

    Returns:
        A list of pandas DataFrames.
    """
    splits = split_result_of_axis_func_pandas(axis, num_splits, df)
    if not isinstance(splits, list):
        splits = [splits]
    return splits


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
def _read_csv_with_offset_pandas_on_ray(
    fname, num_splits, start, end, kwargs, header
):  # pragma: no cover
    """Use a Ray task to read a chunk of a CSV into a Pandas DataFrame.

    Note: Ray functions are not detected by codecov (thus pragma: no cover)

    Args:
        fname: The filename of the file to open.
        num_splits: The number of splits (partitions) to separate the DataFrame into.
        start: The start byte offset.
        end: The end byte offset.
        kwargs: The kwargs for the Pandas `read_csv` function.
        header: The header of the file.

    Returns:
         A list containing the split Pandas DataFrames and the Index as the last
            element. If there is not `index_col` set, then we just return the length.
            This is used to determine the total length of the DataFrame to build a
            default Index.
    """
    index_col = kwargs.get("index_col", None)
    # pop "compression" from kwargs because bio is uncompressed
    bio = file_open(fname, "rb", kwargs.pop("compression", "infer"))
    bio.seek(start)
    to_read = header + bio.read(end - start)
    bio.close()
    pandas_df = pandas.read_csv(BytesIO(to_read), **kwargs)
    pandas_df.columns = pandas.RangeIndex(len(pandas_df.columns))
    if index_col is not None:
        index = pandas_df.index
        # Partitions must have RangeIndex
        pandas_df.index = pandas.RangeIndex(0, len(pandas_df))
    else:
        # We will use the lengths to build the index if we are not given an
        # `index_col`.
        index = len(pandas_df)
    return _split_result_for_readers(1, num_splits, pandas_df) + [
        index,
        pandas_df.dtypes,
    ]


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


class PandasOnRayIO(RayIO):

    frame_mgr_cls = PandasOnRayFrameManager
    frame_partition_cls = PandasOnRayFramePartition
    query_compiler_cls = PandasQueryCompiler

    read_parquet_remote_task = _read_parquet_columns
    read_csv_remote_task = _read_csv_with_offset_pandas_on_ray
    read_hdf_remote_task = _read_hdf_columns
    read_feather_remote_task = _read_feather_columns
    read_sql_remote_task = _read_sql_with_limit_offset
