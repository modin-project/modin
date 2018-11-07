from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas
from pandas.io.common import _infer_compression

import inspect
from io import BytesIO
import os
import py
import ray
import re
import warnings
import numpy as np

from .dataframe import DataFrame
from .utils import from_pandas
from ..data_management.partitioning.partition_collections import RayBlockPartitions
from ..data_management.partitioning.remote_partition import PandasOnRayRemotePartition
from ..data_management.partitioning.axis_partition import (
    split_result_of_axis_func_pandas,
)
from modin.data_management.query_compiler import PandasQueryCompiler

PQ_INDEX_REGEX = re.compile("__index_level_\d+__")  # noqa W605


# Parquet
def read_parquet(path, engine="auto", columns=None, **kwargs):
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
    return _read_parquet_pandas_on_ray(path, engine, columns, **kwargs)


def _read_parquet_pandas_on_ray(path, engine, columns, **kwargs):
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
    df = DataFrame(query_compiler=new_manager)
    return df


# CSV
def _skip_header(f, kwargs={}):
    lines_read = 0
    comment = kwargs.get("comment", None)
    skiprows = kwargs.get("skiprows", None)
    encoding = kwargs.get("encoding", None)
    header = kwargs.get("header", "infer")
    names = kwargs.get("names", None)

    if header is None:
        return lines_read
    elif header == "infer":
        if names is not None:
            return lines_read
        else:
            header = 0
    # Skip lines before the header
    if isinstance(skiprows, int):
        lines_read += skiprows
        for _ in range(skiprows):
            f.readline()
        skiprows = None

    header_lines = header + 1 if isinstance(header, int) else max(header) + 1
    header_lines_skipped = 0
    # Python 2 files use a read-ahead buffer which breaks our use of tell()
    for line in iter(f.readline, ""):
        lines_read += 1
        skip = False
        if not skip and comment is not None:
            if encoding is not None:
                skip |= line.decode(encoding)[0] == comment
            else:
                skip |= line.decode()[0] == comment
        if not skip and callable(skiprows):
            skip |= skiprows(lines_read)
        elif not skip and hasattr(skiprows, "__contains__"):
            skip |= lines_read in skiprows

        if not skip:
            header_lines_skipped += 1
            if header_lines_skipped == header_lines:
                return lines_read
    return lines_read


def _read_csv_from_file_pandas_on_ray(filepath, kwargs={}):
    """Constructs a DataFrame from a CSV file.

    Args:
        filepath (str): path to the CSV file.
        npartitions (int): number of partitions for the DataFrame.
        kwargs (dict): args excluding filepath provided to read_csv.

    Returns:
        DataFrame or Series constructed from CSV file.
    """
    empty_pd_df = pandas.read_csv(filepath, **dict(kwargs, nrows=0, skipfooter=0))
    column_names = empty_pd_df.columns
    skipfooter = kwargs.get("skipfooter", None)
    partition_kwargs = dict(kwargs, header=None, names=column_names, skipfooter=0)
    with open(filepath, "rb") as f:
        # Get the BOM if necessary
        prefix = b""
        if kwargs.get("encoding", None) is not None:
            prefix = f.readline()
            partition_kwargs["skiprows"] = 1
            f.seek(0, os.SEEK_SET)  # Return to beginning of file

        prefix_id = ray.put(prefix)
        partition_kwargs_id = ray.put(partition_kwargs)
        # Skip the header since we already have the header information
        _skip_header(f, kwargs)
        # Launch tasks to read partitions
        partition_ids = []
        index_ids = []
        total_bytes = os.path.getsize(filepath)
        # Max number of partitions available
        num_parts = RayBlockPartitions._compute_num_partitions()
        # This is the number of splits for the columns
        num_splits = min(len(column_names), num_parts)
        # This is the chunksize each partition will read
        chunk_size = max(1, (total_bytes - f.tell()) // num_parts)

        while f.tell() < total_bytes:
            start = f.tell()
            f.seek(chunk_size, os.SEEK_CUR)
            f.readline()  # Read a whole number of lines
            partition_id = _read_csv_with_offset_pandas_on_ray._submit(
                args=(
                    filepath,
                    num_splits,
                    start,
                    f.tell(),
                    partition_kwargs_id,
                    prefix_id,
                ),
                num_return_vals=num_splits + 1,
            )
            partition_ids.append(
                [PandasOnRayRemotePartition(obj) for obj in partition_id[:-1]]
            )
            index_ids.append(partition_id[-1])

    index_col = kwargs.get("index_col", None)
    if index_col is None:
        new_index = pandas.RangeIndex(sum(ray.get(index_ids)))
    else:
        new_index_ids = get_index.remote([empty_pd_df.index.name], *index_ids)
        new_index = ray.get(new_index_ids)

    new_manager = PandasQueryCompiler(
        RayBlockPartitions(np.array(partition_ids)), new_index, column_names
    )
    df = DataFrame(query_compiler=new_manager)

    if skipfooter:
        df = df.drop(df.index[-skipfooter:])
    if kwargs.get("squeeze", False) and len(df.columns) == 1:
        return df[df.columns[0]]

    return df


def _read_csv_from_pandas(filepath_or_buffer, kwargs):
    pd_obj = pandas.read_csv(filepath_or_buffer, **kwargs)
    if isinstance(pd_obj, pandas.DataFrame):
        return from_pandas(pd_obj)
    elif isinstance(pd_obj, pandas.io.parsers.TextFileReader):
        # Overwriting the read method should return a ray DataFrame for calls
        # to __next__ and get_chunk
        pd_read = pd_obj.read
        pd_obj.read = lambda *args, **kwargs: from_pandas(pd_read(*args, **kwargs))
    return pd_obj


def read_csv(
    filepath_or_buffer,
    sep=",",
    delimiter=None,
    header="infer",
    names=None,
    index_col=None,
    usecols=None,
    squeeze=False,
    prefix=None,
    mangle_dupe_cols=True,
    dtype=None,
    engine=None,
    converters=None,
    true_values=None,
    false_values=None,
    skipinitialspace=False,
    skiprows=None,
    nrows=None,
    na_values=None,
    keep_default_na=True,
    na_filter=True,
    verbose=False,
    skip_blank_lines=True,
    parse_dates=False,
    infer_datetime_format=False,
    keep_date_col=False,
    date_parser=None,
    dayfirst=False,
    iterator=False,
    chunksize=None,
    compression="infer",
    thousands=None,
    decimal=b".",
    lineterminator=None,
    quotechar='"',
    quoting=0,
    escapechar=None,
    comment=None,
    encoding=None,
    dialect=None,
    tupleize_cols=None,
    error_bad_lines=True,
    warn_bad_lines=True,
    skipfooter=0,
    doublequote=True,
    delim_whitespace=False,
    low_memory=True,
    memory_map=False,
    float_precision=None,
):
    """Read csv file from local disk.
    Args:
        filepath:
              The filepath of the csv file.
              We only support local files for now.
        kwargs: Keyword arguments in pandas::from_csv
    """
    # The intention of the inspection code is to reduce the amount of
    # communication we have to do between processes and nodes. We take a quick
    # pass over the arguments and remove those that are default values so we
    # don't have to serialize and send them to the workers. Because the
    # arguments list is so long, this does end up saving time based on the
    # number of nodes in the cluster.
    frame = inspect.currentframe()
    _, _, _, kwargs = inspect.getargvalues(frame)
    try:
        args, _, _, defaults, _, _, _ = inspect.getfullargspec(read_csv)
        defaults = dict(zip(args[1:], defaults))
        kwargs = {
            kw: kwargs[kw]
            for kw in kwargs
            if kw in defaults and kwargs[kw] != defaults[kw]
        }
    # This happens on Python2, we will just default to serializing the entire dictionary
    except AttributeError:
        # We suppress the error and delete the kwargs not needed in the remote function.
        del kwargs["filepath_or_buffer"]
        del kwargs["frame"]

    if isinstance(filepath_or_buffer, str):
        if not os.path.exists(filepath_or_buffer):
            warnings.warn(
                "File not found on disk. Defaulting to Pandas implementation.",
                UserWarning,
            )
            return _read_csv_from_pandas(filepath_or_buffer, kwargs)
    elif not isinstance(filepath_or_buffer, py.path.local):
        read_from_pandas = True
        # Pandas read_csv supports pathlib.Path
        try:
            import pathlib

            if isinstance(filepath_or_buffer, pathlib.Path):
                read_from_pandas = False
        except ImportError:
            pass
        if read_from_pandas:
            warnings.warn(
                "Reading from buffer. Defaulting to Pandas implementation.", UserWarning
            )
            return _read_csv_from_pandas(filepath_or_buffer, kwargs)
    if _infer_compression(filepath_or_buffer, compression) is not None:
        warnings.warn(
            "Compression detected. Defaulting to Pandas implementation.", UserWarning
        )
        return _read_csv_from_pandas(filepath_or_buffer, kwargs)
    if chunksize is not None:
        warnings.warn(
            "Reading chunks from a file. Defaulting to Pandas implementation.",
            UserWarning,
        )
        return _read_csv_from_pandas(filepath_or_buffer, kwargs)
    if skiprows is not None and not isinstance(skiprows, int):
        warnings.warn(
            (
                "Defaulting to Pandas implementation. To speed up "
                "read_csv through the Modin implementation, "
                "comment the rows to skip instead."
            )
        )
        return _read_csv_from_pandas(filepath_or_buffer, kwargs)
    # TODO: replace this by reading lines from file.
    if nrows is not None:
        warnings.warn("Defaulting to Pandas implementation.", UserWarning)
        return _read_csv_from_pandas(filepath_or_buffer, kwargs)
    else:
        return _read_csv_from_file_pandas_on_ray(filepath_or_buffer, kwargs)


def read_json(
    path_or_buf=None,
    orient=None,
    typ="frame",
    dtype=True,
    convert_axes=True,
    convert_dates=True,
    keep_default_dates=True,
    numpy=False,
    precise_float=False,
    date_unit=None,
    encoding=None,
    lines=False,
    chunksize=None,
    compression="infer",
):
    warnings.warn("Defaulting to Pandas implementation", UserWarning)
    port_frame = pandas.read_json(
        path_or_buf,
        orient,
        typ,
        dtype,
        convert_axes,
        convert_dates,
        keep_default_dates,
        numpy,
        precise_float,
        date_unit,
        encoding,
        lines,
        chunksize,
        compression,
    )
    ray_frame = from_pandas(port_frame)
    return ray_frame


def read_gbq(
    query,
    project_id=None,
    index_col=None,
    col_order=None,
    reauth=False,
    verbose=None,
    private_key=None,
    dialect="legacy",
    **kwargs
):
    warnings.warn("Defaulting to Pandas implementation", UserWarning)
    port_frame = pandas.read_gbq(
        query,
        project_id=project_id,
        index_col=index_col,
        col_order=col_order,
        reauth=reauth,
        verbose=verbose,
        private_key=private_key,
        dialect=dialect,
        **kwargs
    )
    ray_frame = from_pandas(port_frame)
    return ray_frame


def read_html(
    io,
    match=".+",
    flavor=None,
    header=None,
    index_col=None,
    skiprows=None,
    attrs=None,
    parse_dates=False,
    tupleize_cols=None,
    thousands=",",
    encoding=None,
    decimal=".",
    converters=None,
    na_values=None,
    keep_default_na=True,
):
    warnings.warn("Defaulting to Pandas implementation", UserWarning)
    port_frame = pandas.read_html(
        io,
        match,
        flavor,
        header,
        index_col,
        skiprows,
        attrs,
        parse_dates,
        tupleize_cols,
        thousands,
        encoding,
        decimal,
        converters,
        na_values,
        keep_default_na,
    )
    ray_frame = from_pandas(port_frame[0])
    return ray_frame


def read_clipboard(sep=r"\s+"):
    warnings.warn("Defaulting to Pandas implementation", UserWarning)
    port_frame = pandas.read_clipboard(sep)
    ray_frame = from_pandas(port_frame)
    return ray_frame


def read_excel(
    io,
    sheet_name=0,
    header=0,
    skiprows=None,
    index_col=None,
    names=None,
    usecols=None,
    parse_dates=False,
    date_parser=None,
    na_values=None,
    thousands=None,
    convert_float=True,
    converters=None,
    dtype=None,
    true_values=None,
    false_values=None,
    engine=None,
    squeeze=False,
):
    warnings.warn("Defaulting to Pandas implementation", UserWarning)
    port_frame = pandas.read_excel(
        io,
        sheet_name,
        header,
        skiprows,
        index_col,
        names,
        usecols,
        parse_dates,
        date_parser,
        na_values,
        thousands,
        convert_float,
        converters,
        dtype,
        true_values,
        false_values,
        engine,
        squeeze,
    )
    ray_frame = from_pandas(port_frame)
    return ray_frame


def read_hdf(path_or_buf, key=None, mode="r"):
    warnings.warn("Defaulting to Pandas implementation", UserWarning)
    port_frame = pandas.read_hdf(path_or_buf, key, mode)
    ray_frame = from_pandas(port_frame)
    return ray_frame


def read_feather(path, nthreads=1):
    warnings.warn("Defaulting to Pandas implementation", UserWarning)
    port_frame = pandas.read_feather(path)
    ray_frame = from_pandas(port_frame)
    return ray_frame


def read_msgpack(path_or_buf, encoding="utf-8", iterator=False):
    warnings.warn("Defaulting to Pandas implementation", UserWarning)
    port_frame = pandas.read_msgpack(path_or_buf, encoding, iterator)
    ray_frame = from_pandas(port_frame)
    return ray_frame


def read_stata(
    filepath_or_buffer,
    convert_dates=True,
    convert_categoricals=True,
    encoding=None,
    index_col=None,
    convert_missing=False,
    preserve_dtypes=True,
    columns=None,
    order_categoricals=True,
    chunksize=None,
    iterator=False,
):
    warnings.warn("Defaulting to Pandas implementation", UserWarning)
    port_frame = pandas.read_stata(
        filepath_or_buffer,
        convert_dates,
        convert_categoricals,
        encoding,
        index_col,
        convert_missing,
        preserve_dtypes,
        columns,
        order_categoricals,
        chunksize,
        iterator,
    )
    ray_frame = from_pandas(port_frame)
    return ray_frame


def read_sas(
    filepath_or_buffer,
    format=None,
    index=None,
    encoding=None,
    chunksize=None,
    iterator=False,
):
    warnings.warn("Defaulting to Pandas implementation", UserWarning)
    port_frame = pandas.read_sas(
        filepath_or_buffer, format, index, encoding, chunksize, iterator
    )
    ray_frame = from_pandas(port_frame)
    return ray_frame


def read_pickle(path, compression="infer"):
    warnings.warn("Defaulting to Pandas implementation", UserWarning)
    port_frame = pandas.read_pickle(path, compression)
    ray_frame = from_pandas(port_frame)
    return ray_frame


def read_sql(
    sql,
    con,
    index_col=None,
    coerce_float=True,
    params=None,
    parse_dates=None,
    columns=None,
    chunksize=None,
):
    warnings.warn("Defaulting to Pandas implementation", UserWarning)
    port_frame = pandas.read_sql(
        sql, con, index_col, coerce_float, params, parse_dates, columns, chunksize
    )
    ray_frame = from_pandas(port_frame)
    return ray_frame


@ray.remote
def get_index(index_name, *partition_indices):
    index = partition_indices[0].append(partition_indices[1:])
    index.names = index_name
    return index


@ray.remote
def _read_csv_with_offset_pandas_on_ray(fname, num_splits, start, end, kwargs, header):
    """Use a Ray task to read a chunk of a CSV into a Pandas DataFrame.

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
    bio = open(fname, "rb")
    bio.seek(start)
    to_read = header + bio.read(end - start)
    bio.close()
    pandas_df = pandas.read_csv(BytesIO(to_read), **kwargs)
    pandas_df.columns = pandas.RangeIndex(len(pandas_df.columns))
    if kwargs.get("index_col", None) is not None:
        index = pandas_df.index
        # Partitions must have RangeIndex
        pandas_df.index = pandas.RangeIndex(0, len(pandas_df))
    else:
        # We will use the lengths to build the index if we are not given an
        # `index_col`.
        index = len(pandas_df)
    return split_result_of_axis_func_pandas(1, num_splits, pandas_df) + [index]


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
