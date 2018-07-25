from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas
from pandas.io.common import _infer_compression

from io import BytesIO
import os
import py
from pyarrow.parquet import ParquetFile
import pyarrow.parquet as pq
import re
import warnings
import numpy as np

from .dataframe import ray, DataFrame
from . import get_npartitions
from .utils import from_pandas, _partition_pandas_dataframe

PQ_INDEX_REGEX = re.compile('__index_level_\d+__')


# Parquet
def read_parquet(path, engine='auto', columns=None, **kwargs):
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
    if not columns:
        pf = ParquetFile(path)
        columns = [
            name for name in pf.metadata.schema.names
            if not PQ_INDEX_REGEX.match(name)
        ]

    # Each item in this list will be a column of original df
    # partitioned to smaller pieces along rows.
    # We need to transpose the oids array to fit our schema.
    blk_partitions = [
        ray.get(_read_parquet_column.remote(path, col, kwargs))
        for col in columns
    ]
    blk_partitions = np.array(blk_partitions).T

    return DataFrame(block_partitions=blk_partitions, columns=columns)


# CSV
def _skip_header(f, kwargs={}):
    lines_read = 0
    comment = kwargs["comment"]
    skiprows = kwargs["skiprows"]
    encoding = kwargs["encoding"]
    header = kwargs["header"]
    names = kwargs["names"]

    if header is None:
        return lines_read

    if header == "infer":
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


def _read_csv_from_file(filepath, npartitions, kwargs={}):
    """Constructs a DataFrame from a CSV file.

    Args:
        filepath (str): path to the CSV file.
        npartitions (int): number of partitions for the DataFrame.
        kwargs (dict): args excluding filepath provided to read_csv.

    Returns:
        DataFrame or Series constructed from CSV file.
    """
    empty_pd_df = pandas.read_csv(
        filepath, **dict(kwargs, nrows=0, skipfooter=0, skip_footer=0))
    names = empty_pd_df.columns

    skipfooter = kwargs["skipfooter"]
    skip_footer = kwargs["skip_footer"]

    partition_kwargs = dict(
        kwargs, header=None, names=names, skipfooter=0, skip_footer=0)
    with open(filepath, "rb") as f:
        # Get the BOM if necessary
        prefix = b""
        if kwargs["encoding"] is not None:
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
        chunk_size = max(1, (total_bytes - f.tell()) // npartitions)
        while f.tell() < total_bytes:
            start = f.tell()
            f.seek(chunk_size, os.SEEK_CUR)
            f.readline()  # Read a whole number of lines

            if f.tell() >= total_bytes:
                kwargs["skipfooter"] = skipfooter
                kwargs["skip_footer"] = skip_footer

            partition_id, index_id = _read_csv_with_offset._submit(
                args=(filepath, start, f.tell(), partition_kwargs_id,
                      prefix_id),
                num_return_vals=2)
            partition_ids.append(partition_id)
            index_ids.append(index_id)

    # Construct index
    index_id = get_index.remote([empty_pd_df.index.name], *index_ids) \
        if kwargs["index_col"] is not None else None

    df = DataFrame(row_partitions=partition_ids, columns=names, index=index_id)

    skipfooter = kwargs["skipfooter"] or kwargs["skip_footer"]
    if skipfooter:
        df = df.drop(df.index[-skipfooter:])
    if kwargs["squeeze"] and len(df.columns) == 1:
        return df[df.columns[0]]

    return df


def _read_csv_from_pandas(filepath_or_buffer, kwargs):
    pd_obj = pandas.read_csv(filepath_or_buffer, **kwargs)

    if isinstance(pd_obj, pandas.DataFrame):
        return from_pandas(pd_obj, get_npartitions())
    elif isinstance(pd_obj, pandas.io.parsers.TextFileReader):
        # Overwriting the read method should return a ray DataFrame for calls
        # to __next__ and get_chunk
        pd_read = pd_obj.read
        pd_obj.read = lambda *args, **kwargs: \
            from_pandas(pd_read(*args, **kwargs), get_npartitions())

    return pd_obj


def read_csv(filepath_or_buffer,
             sep=',',
             delimiter=None,
             header='infer',
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
             compression='infer',
             thousands=None,
             decimal=b'.',
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
             skip_footer=0,
             doublequote=True,
             delim_whitespace=False,
             as_recarray=None,
             compact_ints=None,
             use_unsigned=None,
             low_memory=True,
             buffer_lines=None,
             memory_map=False,
             float_precision=None):
    """Read csv file from local disk.
    Args:
        filepath:
              The filepath of the csv file.
              We only support local files for now.
        kwargs: Keyword arguments in pandas::from_csv
    """

    kwargs = {
        'sep': sep,
        'delimiter': delimiter,
        'header': header,
        'names': names,
        'index_col': index_col,
        'usecols': usecols,
        'squeeze': squeeze,
        'prefix': prefix,
        'mangle_dupe_cols': mangle_dupe_cols,
        'dtype': dtype,
        'engine': engine,
        'converters': converters,
        'true_values': true_values,
        'false_values': false_values,
        'skipinitialspace': skipinitialspace,
        'skiprows': skiprows,
        'nrows': nrows,
        'na_values': na_values,
        'keep_default_na': keep_default_na,
        'na_filter': na_filter,
        'verbose': verbose,
        'skip_blank_lines': skip_blank_lines,
        'parse_dates': parse_dates,
        'infer_datetime_format': infer_datetime_format,
        'keep_date_col': keep_date_col,
        'date_parser': date_parser,
        'dayfirst': dayfirst,
        'iterator': iterator,
        'chunksize': chunksize,
        'compression': compression,
        'thousands': thousands,
        'decimal': decimal,
        'lineterminator': lineterminator,
        'quotechar': quotechar,
        'quoting': quoting,
        'escapechar': escapechar,
        'comment': comment,
        'encoding': encoding,
        'dialect': dialect,
        'tupleize_cols': tupleize_cols,
        'error_bad_lines': error_bad_lines,
        'warn_bad_lines': warn_bad_lines,
        'skipfooter': skipfooter,
        'skip_footer': skip_footer,
        'doublequote': doublequote,
        'delim_whitespace': delim_whitespace,
        'as_recarray': as_recarray,
        'compact_ints': compact_ints,
        'use_unsigned': use_unsigned,
        'low_memory': low_memory,
        'buffer_lines': buffer_lines,
        'memory_map': memory_map,
        'float_precision': float_precision,
    }

    if isinstance(filepath_or_buffer, str):
        if not os.path.exists(filepath_or_buffer):
            warnings.warn(("File not found on disk. "
                           "Defaulting to Pandas implementation."),
                          PendingDeprecationWarning)

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
            warnings.warn(("Reading from buffer. "
                           "Defaulting to Pandas implementation."),
                          PendingDeprecationWarning)

            return _read_csv_from_pandas(filepath_or_buffer, kwargs)

    if _infer_compression(filepath_or_buffer, compression) is not None:
        warnings.warn(("Compression detected. "
                       "Defaulting to Pandas implementation."),
                      PendingDeprecationWarning)

        return _read_csv_from_pandas(filepath_or_buffer, kwargs)

    if as_recarray:
        warnings.warn("Defaulting to Pandas implementation.",
                      PendingDeprecationWarning)

        return _read_csv_from_pandas(filepath_or_buffer, kwargs)

    if chunksize is not None:
        warnings.warn(("Reading chunks from a file. "
                       "Defaulting to Pandas implementation."),
                      PendingDeprecationWarning)

        return _read_csv_from_pandas(filepath_or_buffer, kwargs)

    if skiprows is not None and not isinstance(skiprows, int):
        warnings.warn(("Defaulting to Pandas implementation. To speed up "
                       "read_csv through the Pandas on Ray implementation, "
                       "comment the rows to skip instead."))

        return _read_csv_from_pandas(filepath_or_buffer, kwargs)

    # TODO: replace this by reading lines from file.
    if nrows is not None:
        warnings.warn("Defaulting to Pandas implementation.",
                      PendingDeprecationWarning)

        return _read_csv_from_pandas(filepath_or_buffer, kwargs)

    return _read_csv_from_file(filepath_or_buffer, get_npartitions(), kwargs)


def read_json(path_or_buf=None,
              orient=None,
              typ='frame',
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
              compression='infer'):

    warnings.warn("Defaulting to Pandas implementation",
                  PendingDeprecationWarning)

    port_frame = pandas.read_json(
        path_or_buf, orient, typ, dtype, convert_axes, convert_dates,
        keep_default_dates, numpy, precise_float, date_unit, encoding, lines,
        chunksize, compression)
    ray_frame = from_pandas(port_frame, get_npartitions())

    return ray_frame


def read_html(io,
              match='.+',
              flavor=None,
              header=None,
              index_col=None,
              skiprows=None,
              attrs=None,
              parse_dates=False,
              tupleize_cols=None,
              thousands=',',
              encoding=None,
              decimal='.',
              converters=None,
              na_values=None,
              keep_default_na=True):

    warnings.warn("Defaulting to Pandas implementation",
                  PendingDeprecationWarning)

    port_frame = pandas.read_html(io, match, flavor, header, index_col,
                                  skiprows, attrs, parse_dates, tupleize_cols,
                                  thousands, encoding, decimal, converters,
                                  na_values, keep_default_na)
    ray_frame = from_pandas(port_frame[0], get_npartitions())

    return ray_frame


def read_clipboard(sep=r'\s+'):

    warnings.warn("Defaulting to Pandas implementation",
                  PendingDeprecationWarning)

    port_frame = pandas.read_clipboard(sep)
    ray_frame = from_pandas(port_frame, get_npartitions())

    return ray_frame


def read_excel(io,
               sheet_name=0,
               header=0,
               skiprows=None,
               skip_footer=0,
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
               squeeze=False):

    warnings.warn("Defaulting to Pandas implementation",
                  PendingDeprecationWarning)

    port_frame = pandas.read_excel(
        io, sheet_name, header, skiprows, skip_footer, index_col, names,
        usecols, parse_dates, date_parser, na_values, thousands, convert_float,
        converters, dtype, true_values, false_values, engine, squeeze)
    ray_frame = from_pandas(port_frame, get_npartitions())

    return ray_frame


def read_hdf(path_or_buf, key=None, mode='r'):

    warnings.warn("Defaulting to Pandas implementation",
                  PendingDeprecationWarning)

    port_frame = pandas.read_hdf(path_or_buf, key, mode)
    ray_frame = from_pandas(port_frame, get_npartitions())

    return ray_frame


def read_feather(path, nthreads=1):

    warnings.warn("Defaulting to Pandas implementation",
                  PendingDeprecationWarning)

    port_frame = pandas.read_feather(path)
    ray_frame = from_pandas(port_frame, get_npartitions())

    return ray_frame


def read_msgpack(path_or_buf, encoding='utf-8', iterator=False):

    warnings.warn("Defaulting to Pandas implementation",
                  PendingDeprecationWarning)

    port_frame = pandas.read_msgpack(path_or_buf, encoding, iterator)
    ray_frame = from_pandas(port_frame, get_npartitions())

    return ray_frame


def read_stata(filepath_or_buffer,
               convert_dates=True,
               convert_categoricals=True,
               encoding=None,
               index_col=None,
               convert_missing=False,
               preserve_dtypes=True,
               columns=None,
               order_categoricals=True,
               chunksize=None,
               iterator=False):

    warnings.warn("Defaulting to Pandas implementation",
                  PendingDeprecationWarning)

    port_frame = pandas.read_stata(filepath_or_buffer, convert_dates,
                                   convert_categoricals, encoding, index_col,
                                   convert_missing, preserve_dtypes, columns,
                                   order_categoricals, chunksize, iterator)
    ray_frame = from_pandas(port_frame, get_npartitions())

    return ray_frame


def read_sas(filepath_or_buffer,
             format=None,
             index=None,
             encoding=None,
             chunksize=None,
             iterator=False):

    warnings.warn("Defaulting to Pandas implementation",
                  PendingDeprecationWarning)

    port_frame = pandas.read_sas(filepath_or_buffer, format, index, encoding,
                                 chunksize, iterator)
    ray_frame = from_pandas(port_frame, get_npartitions())

    return ray_frame


def read_pickle(path, compression='infer'):

    warnings.warn("Defaulting to Pandas implementation",
                  PendingDeprecationWarning)

    port_frame = pandas.read_pickle(path, compression)
    ray_frame = from_pandas(port_frame, get_npartitions())

    return ray_frame


def read_sql(sql,
             con,
             index_col=None,
             coerce_float=True,
             params=None,
             parse_dates=None,
             columns=None,
             chunksize=None):

    warnings.warn("Defaulting to Pandas implementation",
                  PendingDeprecationWarning)

    port_frame = pandas.read_sql(sql, con, index_col, coerce_float, params,
                                 parse_dates, columns, chunksize)
    ray_frame = from_pandas(port_frame, get_npartitions())

    return ray_frame


@ray.remote
def get_index(index_name, *partition_indices):
    index = partition_indices[0].append(partition_indices[1:])
    index.names = index_name
    return index


@ray.remote
def _read_csv_with_offset(fn, start, end, kwargs={}, header=b''):
    bio = open(fn, 'rb')
    bio.seek(start)
    to_read = header + bio.read(end - start)
    bio.close()
    pandas_df = pandas.read_csv(BytesIO(to_read), **kwargs)
    index = pandas_df.index
    # Partitions must have RangeIndex
    pandas_df.index = pandas.RangeIndex(0, len(pandas_df))
    return pandas_df, index


@ray.remote
def _read_parquet_column(path, column, kwargs={}):
    df = pq.read_pandas(path, columns=[column], **kwargs).to_pandas()
    oids = _partition_pandas_dataframe(df, num_partitions=get_npartitions())
    return oids
