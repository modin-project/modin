from io import BytesIO
import ray
import pyarrow
import pyarrow.csv

from modin.data_management.query_compiler import GandivaQueryCompiler
from modin.engines.base.io import BaseIO
from modin.engines.ray.pandas_on_ray.io import PandasOnRayIO
from .block_partitions import RayBlockPartitions


@ray.remote
def _read_csv_with_offset_pyarrow(
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
    # index_col = kwargs.pop("index_col", None)
    index_col = None
    bio = open(fname, "rb")
    bio.seek(start)
    to_read = header + bio.read(end - start)
    bio.close()
    # pandas_df = pandas.read_csv(BytesIO(to_read), **kwargs)
    pandas_df = pyarrow.csv.read_csv(BytesIO(to_read))
    pandas_df.columns = pandas.RangeIndex(len(pandas_df.columns))
    # We will use the lengths to build the index if we are not given an
    # `index_col`.
    return [pandas_df] + [len(pandas_df)]


class GandivaOnRayIO(PandasOnRayIO):

    block_partitions_cls = RayBlockPartitions
    query_compiler_cls = GandivaQueryCompiler
    read_csv_func = _read_csv_with_offset_pyarrow
