from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from io import BytesIO

import ray
import pandas
import pyarrow as pa
import pyarrow.csv as csv

from modin.backends.pyarrow.query_compiler import PyarrowQueryCompiler
from modin.data_management.utils import get_default_chunksize
from modin.engines.ray.generic.io import RayIO
from modin.experimental.engines.pyarrow_on_ray.frame.partition_manager import (
    PyarrowOnRayFrameManager,
)
from modin.experimental.engines.pyarrow_on_ray.frame.partition import (
    PyarrowOnRayFramePartition,
)


@ray.remote
def _read_csv_with_offset_pyarrow_on_ray(
    fname, num_splits, start, end, kwargs, header
):  # pragma: no cover
    """Use a Ray task to read a chunk of a CSV into a pyarrow Table.
     Note: Ray functions are not detected by codecov (thus pragma: no cover)
     Args:
        fname: The filename of the file to open.
        num_splits: The number of splits (partitions) to separate the DataFrame into.
        start: The start byte offset.
        end: The end byte offset.
        kwargs: The kwargs for the pyarrow `read_csv` function.
        header: The header of the file.
     Returns:
         A list containing the split pyarrow Tables and the the number of
         rows of the tables as the last element. This is used to determine
         the total length of the DataFrame to build a default Index.
    """
    bio = open(fname, "rb")
    # The header line for the CSV file
    first_line = bio.readline()
    bio.seek(start)
    to_read = header + first_line + bio.read(end - start)
    bio.close()
    table = csv.read_csv(
        BytesIO(to_read), parse_options=csv.ParseOptions(header_rows=1)
    )
    chunksize = get_default_chunksize(table.num_columns, num_splits)
    chunks = [
        pa.Table.from_arrays(table.columns[chunksize * i : chunksize * (i + 1)])
        for i in range(num_splits)
    ]
    return chunks + [
        table.num_rows,
        pandas.Series(
            [t.to_pandas_dtype() for t in table.schema.types], index=table.schema.names
        ),
    ]


class PyarrowOnRayIO(RayIO):

    frame_mgr_cls = PyarrowOnRayFrameManager
    frame_partition_cls = PyarrowOnRayFramePartition
    query_compiler_cls = PyarrowQueryCompiler

    read_parquet_remote_task = None
    read_csv_remote_task = _read_csv_with_offset_pyarrow_on_ray
    read_hdf_remote_task = None
    read_feather_remote_task = None
    read_sql_remote_task = None
