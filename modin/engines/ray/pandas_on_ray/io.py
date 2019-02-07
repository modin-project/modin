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
import numpy as np

from modin.error_message import ErrorMessage
from modin.data_management.utils import split_result_of_axis_func_pandas
from modin.data_management.query_compiler import PandasQueryCompiler
from modin.engines.base.io import BaseIO
from .block_partitions import RayBlockPartitions
from .remote_partition import PandasOnRayRemotePartition

PQ_INDEX_REGEX = re.compile("__index_level_\d+__")  # noqa W605


class PandasOnRayIO(BaseIO):

    block_partitions_cls = RayBlockPartitions
    query_compiler_cls = PandasQueryCompiler

    @classmethod
    def read_parquet(cls, path, engine, columns, **kwargs):
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
                name
                for name in pf.metadata.schema.names
                if not PQ_INDEX_REGEX.match(name)
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
            columns[i : i + column_splits]
            for i in range(0, len(columns), column_splits)
        ]
        # Each item in this list will be a list of columns of original df
        # partitioned to smaller pieces along rows.
        # We need to transpose the oids array to fit our schema.
        blk_partitions = np.array(
            [
                _read_parquet_columns._remote(
                    args=(path, cols, num_splits, kwargs),
                    num_return_vals=num_splits + 1,
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
        new_query_compiler = PandasQueryCompiler(
            RayBlockPartitions(remote_partitions), index, columns
        )

        return new_query_compiler

    # CSV
    @classmethod
    def _skip_header(cls, f, kwargs={}):
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

    @classmethod
    def _read_csv_from_file_pandas_on_ray(cls, filepath, kwargs={}):
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
        skiprows = kwargs.pop("skiprows", None)
        partition_kwargs = dict(
            kwargs, header=None, names=column_names, skipfooter=0, skiprows=None
        )
        with open(filepath, "rb") as f:
            # Get the BOM if necessary
            prefix = b""
            if kwargs.get("encoding", None) is not None:
                prefix = f.readline()
                partition_kwargs["skiprows"] = 1
                f.seek(0, os.SEEK_SET)  # Return to beginning of file

            prefix_id = ray.put(prefix)
            partition_kwargs_id = ray.put(partition_kwargs)
            # Skip the header since we already have the header information and skip the
            # rows we are told to skip.
            kwargs["skiprows"] = skiprows
            cls._skip_header(f, kwargs)
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
                partition_id = _read_csv_with_offset_pandas_on_ray._remote(
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

        new_query_compiler = PandasQueryCompiler(
            RayBlockPartitions(np.array(partition_ids)), new_index, column_names
        )

        if skipfooter:
            new_query_compiler = new_query_compiler.drop(
                new_query_compiler.index[-skipfooter:]
            )
        if kwargs.get("squeeze", False) and len(new_query_compiler.columns) == 1:
            return new_query_compiler[new_query_compiler.columns[0]]
        return new_query_compiler

    @classmethod
    def _read_csv_from_pandas(cls, filepath_or_buffer, kwargs):
        pd_obj = pandas.read_csv(filepath_or_buffer, **kwargs)
        if isinstance(pd_obj, pandas.DataFrame):
            return cls.from_pandas(pd_obj)
        elif isinstance(pd_obj, pandas.io.parsers.TextFileReader):
            # Overwriting the read method should return a ray DataFrame for calls
            # to __next__ and get_chunk
            pd_read = pd_obj.read
            pd_obj.read = lambda *args, **kwargs: cls.from_pandas(
                pd_read(*args, **kwargs)
            )
        return pd_obj

    @classmethod
    def read_csv(
        cls,
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
        kwargs = {
            "filepath_or_buffer": filepath_or_buffer,
            "sep": sep,
            "delimiter": delimiter,
            "header": header,
            "names": names,
            "index_col": index_col,
            "usecols": usecols,
            "squeeze": squeeze,
            "prefix": prefix,
            "mangle_dupe_cols": mangle_dupe_cols,
            "dtype": dtype,
            "engine": engine,
            "converters": converters,
            "true_values": true_values,
            "false_values": false_values,
            "skipinitialspace": skipinitialspace,
            "skiprows": skiprows,
            "nrows": nrows,
            "na_values": na_values,
            "keep_default_na": keep_default_na,
            "na_filter": na_filter,
            "verbose": verbose,
            "skip_blank_lines": skip_blank_lines,
            "parse_dates": parse_dates,
            "infer_datetime_format": infer_datetime_format,
            "keep_date_col": keep_date_col,
            "date_parser": date_parser,
            "dayfirst": dayfirst,
            "iterator": iterator,
            "chunksize": chunksize,
            "compression": compression,
            "thousands": thousands,
            "decimal": decimal,
            "lineterminator": lineterminator,
            "quotechar": quotechar,
            "quoting": quoting,
            "escapechar": escapechar,
            "comment": comment,
            "encoding": encoding,
            "dialect": dialect,
            "tupleize_cols": tupleize_cols,
            "error_bad_lines": error_bad_lines,
            "warn_bad_lines": warn_bad_lines,
            "skipfooter": skipfooter,
            "doublequote": doublequote,
            "delim_whitespace": delim_whitespace,
            "low_memory": low_memory,
            "memory_map": memory_map,
            "float_precision": float_precision,
        }
        return cls._read(**kwargs)

    @classmethod
    def _read(cls, filepath_or_buffer, **kwargs):
        """Read csv file from local disk.
        Args:
            filepath_or_buffer:
                  The filepath of the csv file.
                  We only support local files for now.
            kwargs: Keyword arguments in pandas.read_csv
        """
        # The intention of the inspection code is to reduce the amount of
        # communication we have to do between processes and nodes. We take a quick
        # pass over the arguments and remove those that are default values so we
        # don't have to serialize and send them to the workers. Because the
        # arguments list is so long, this does end up saving time based on the
        # number of nodes in the cluster.
        try:
            args, _, _, defaults, _, _, _ = inspect.getfullargspec(cls.read_csv)
            defaults = dict(zip(args[2:], defaults))
            filtered_kwargs = {
                kw: kwargs[kw]
                for kw in kwargs
                if kw in defaults
                and not isinstance(kwargs[kw], type(defaults[kw]))
                or kwargs[kw] != defaults[kw]
            }
        # This happens on Python2, we will just default to serializing the entire dictionary
        except AttributeError:
            filtered_kwargs = kwargs

        if isinstance(filepath_or_buffer, str):
            if not os.path.exists(filepath_or_buffer):
                ErrorMessage.default_to_pandas("File not found on disk")
                return cls._read_csv_from_pandas(filepath_or_buffer, filtered_kwargs)
        elif not isinstance(filepath_or_buffer, py.path.local):
            read_from_pandas = True
            # Pandas read_csv supports pathlib.Path
            try:
                import pathlib

                if isinstance(filepath_or_buffer, pathlib.Path):
                    read_from_pandas = False
            except ImportError:  # pragma: no cover
                pass
            if read_from_pandas:
                ErrorMessage.default_to_pandas("Reading from buffer.")
                return cls._read_csv_from_pandas(filepath_or_buffer, kwargs)
        if (
            _infer_compression(filepath_or_buffer, kwargs.get("compression"))
            is not None
        ):
            ErrorMessage.default_to_pandas("Compression detected.")
            return cls._read_csv_from_pandas(filepath_or_buffer, filtered_kwargs)

        chunksize = kwargs.get("chunksize")
        if chunksize is not None:
            ErrorMessage.default_to_pandas("Reading chunks from a file.")
            return cls._read_csv_from_pandas(filepath_or_buffer, filtered_kwargs)

        skiprows = kwargs.get("skiprows")
        if skiprows is not None and not isinstance(skiprows, int):
            ErrorMessage.default_to_pandas("skiprows parameter not optimized yet.")
            return cls._read_csv_from_pandas(filepath_or_buffer, kwargs)
        # TODO: replace this by reading lines from file.
        if kwargs.get("nrows") is not None:
            ErrorMessage.default_to_pandas("`read_csv` with `nrows`")
            return cls._read_csv_from_pandas(filepath_or_buffer, filtered_kwargs)
        else:
            return cls._read_csv_from_file_pandas_on_ray(
                filepath_or_buffer, filtered_kwargs
            )

    @classmethod
    def _validate_hdf_format(cls, path_or_buf):
        s = pandas.HDFStore(path_or_buf)
        groups = s.groups()
        if len(groups) == 0:
            raise ValueError("No dataset in HDF5 file.")
        candidate_only_group = groups[0]
        format = getattr(candidate_only_group._v_attrs, "table_type", None)
        s.close()
        return format

    @classmethod
    def read_hdf(cls, path_or_buf, **kwargs):
        """Load a h5 file from the file path or buffer, returning a DataFrame.

        Args:
            path_or_buf: string, buffer or path object
                Path to the file to open, or an open :class:`pandas.HDFStore` object.
            kwargs: Pass into pandas.read_hdf function.

        Returns:
            DataFrame constructed from the h5 file.
        """
        format = cls._validate_hdf_format(path_or_buf=path_or_buf)

        if format is None:
            ErrorMessage.default_to_pandas(
                "File format seems to be `fixed`. For better distribution consider saving the file in `table` format. "
                "df.to_hdf(format=`table`)."
            )
            return cls.from_pandas(pandas.read_hdf(path_or_buf=path_or_buf, **kwargs))

        columns = kwargs.get("columns", None)
        if not columns:
            empty_pd_df = pandas.read_hdf(path_or_buf, start=0, stop=0)
            columns = empty_pd_df.columns

        num_partitions = RayBlockPartitions._compute_num_partitions()
        num_splits = min(len(columns), num_partitions)
        # Each item in this list will be a list of column names of the original df
        column_splits = (
            len(columns) // num_partitions
            if len(columns) % num_partitions == 0
            else len(columns) // num_partitions + 1
        )
        col_partitions = [
            columns[i : i + column_splits]
            for i in range(0, len(columns), column_splits)
        ]
        blk_partitions = np.array(
            [
                _read_hdf_columns._remote(
                    args=(path_or_buf, cols, num_splits, kwargs),
                    num_return_vals=num_splits + 1,
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
        new_query_compiler = PandasQueryCompiler(
            RayBlockPartitions(remote_partitions), index, columns
        )
        return new_query_compiler

    @classmethod
    def read_feather(cls, path, nthreads=1, columns=None):
        """Read a pandas.DataFrame from Feather format.
           Ray DataFrame only supports pyarrow engine for now.

        Args:
            path: The filepath of the feather file.
                  We only support local files for now.
            nthreads: since we use pyarrow reader then this does nothing.
                multi threading is set to True by default
            columns: not supported by pandas api, but can be passed here to read only specific columns

        Notes:
            pyarrow feather is used. Please refer to the documentation here
            https://arrow.apache.org/docs/python/api.html#feather-format
        """
        if not columns:
            from pyarrow.feather import FeatherReader

            fr = FeatherReader(path)
            columns = [fr.get_column_name(i) for i in range(fr.num_columns)]

        num_partitions = RayBlockPartitions._compute_num_partitions()
        num_splits = min(len(columns), num_partitions)
        # Each item in this list will be a list of column names of the original df
        column_splits = (
            len(columns) // num_partitions
            if len(columns) % num_partitions == 0
            else len(columns) // num_partitions + 1
        )
        col_partitions = [
            columns[i : i + column_splits]
            for i in range(0, len(columns), column_splits)
        ]
        blk_partitions = np.array(
            [
                _read_feather_columns._remote(
                    args=(path, cols, num_splits), num_return_vals=num_splits + 1
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
        new_query_compiler = PandasQueryCompiler(
            RayBlockPartitions(remote_partitions), index, columns
        )
        return new_query_compiler


@ray.remote
def get_index(index_name, *partition_indices):  # pragma: no cover
    """Get the index from the indices returned by the workers.

    Note: Ray functions are not detected by codecov (thus pragma: no cover)"""
    index = partition_indices[0].append(partition_indices[1:])
    index.names = index_name
    return index


def _split_result_for_readers(axis, num_splits, df):
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
    index_col = kwargs.pop("index_col", None)
    bio = open(fname, "rb")
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
    return _split_result_for_readers(1, num_splits, pandas_df) + [index]


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

    df = pq.read_pandas(path, columns=columns, **kwargs).to_pandas()
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
