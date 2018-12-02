from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas
import ray
import re
import numpy as np

from modin.data_management.partitioning.partition_collections import RayBlockPartitions
from modin.data_management.partitioning.remote_partition import PandasOnRayRemotePartition
from modin.data_management.partitioning.axis_partition import (
    split_result_of_axis_func_pandas,
)
from modin.data_management.query_compiler import PandasQueryCompiler

PQ_INDEX_REGEX = re.compile("__index_level_\d+__")  # noqa W605


def read_parquet(path, engine, columns, **kwargs):
    """Load a parquet object from the file path, returning a DataFrame.
       Ray DataFrame only supports pyarrow engine for now.

    Args:
        path: The filepath of the parquet file.
              We only support local files for now.
        engine: Ray only support pyarrow reader.
                This argument doesn't do anything for now.
        kwargs: Pass into parquet's read_pandas function.

    Notes:
        ParquetFile API is used. Please refer to the documentation here
        https://arrow.apache.org/docs/python/parquet.html
    """

    from pyarrow.parquet import ParquetFile

    if not columns:
        pf = ParquetFile(path)
        columns = [
            name for name in pf.metadata.schema.names if not PQ_INDEX_REGEX.match(name)
        ]
    num_partitions = RayBlockPartitions._compute_num_partitions()
    num_splits = min(len(columns), num_partitions)
    # Each item in this list will be a list of column names of the original df
    column_splits = (
        len(columns) // num_partitions
        if len(columns) % num_partitions == 0
        else len(columns) // num_partitions + 1
    )
    col_partitions = [
        columns[i : i + column_splits] for i in range(0, len(columns), column_splits)
    ]
    # Each item in this list will be a list of columns of original df
    # partitioned to smaller pieces along rows.
    # We need to transpose the oids array to fit our schema.
    blk_partitions = np.array(
        [
            _read_parquet_columns._submit(
                args=(path, cols, num_splits, kwargs), num_return_vals=num_splits + 1
            )
            for cols in col_partitions
        ]
    ).T
    remote_partitions = np.array(
        [
            [PandasOnRayRemotePartition(obj) for obj in row]
            for row in blk_partitions[:-1]
        ]
    )
    index_len = ray.get(blk_partitions[-1][0])
    index = pandas.RangeIndex(index_len)
    new_manager = PandasQueryCompiler(
        RayBlockPartitions(remote_partitions), index, columns
    )

    return new_manager


@ray.remote
def _read_parquet_columns(path, columns, num_splits, kwargs):
    """Use a Ray task to read a column from Parquet into a Pandas DataFrame.

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

    df = pq.read_pandas(path, columns=columns, **kwargs).to_pandas()
    # Append the length of the index here to build it externally
    return split_result_of_axis_func_pandas(0, num_splits, df) + [len(df.index)]
