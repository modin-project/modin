from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas
from pandas.io.common import _infer_compression

import inspect
import os
import py
import ray
import re
import numpy as np
import math

from modin.error_message import ErrorMessage
from modin.engines.base.io import BaseIO

PQ_INDEX_REGEX = re.compile("__index_level_\d+__")  # noqa W605
S3_ADDRESS_REGEX = re.compile("s3://(.*?)/(.*)")


def file_exists(file_path):
    if isinstance(file_path, str):
        match = S3_ADDRESS_REGEX.search(file_path)
        if match:
            import s3fs as S3FS
            from botocore.exceptions import NoCredentialsError

            s3fs = S3FS.S3FileSystem(anon=False)
            try:
                return s3fs.exists(file_path)
            except NoCredentialsError:
                s3fs = S3FS.S3FileSystem(anon=True)
                return s3fs.exists(file_path)
    return os.path.exists(file_path)


def file_open(file_path, mode="rb"):
    if isinstance(file_path, str):
        match = S3_ADDRESS_REGEX.search(file_path)
        if match:
            import s3fs as S3FS
            from botocore.exceptions import NoCredentialsError

            s3fs = S3FS.S3FileSystem(anon=False)
            try:
                return s3fs.open(file_path)
            except NoCredentialsError:
                s3fs = S3FS.S3FileSystem(anon=True)
                return s3fs.open(file_path)
    return open(file_path, mode=mode)


def file_size(f):
    cur_pos = f.tell()
    f.seek(0, os.SEEK_END)
    size = f.tell()
    f.seek(cur_pos, os.SEEK_SET)
    return size


@ray.remote
def get_index(index_name, *partition_indices):  # pragma: no cover
    """Get the index from the indices returned by the workers.

    Note: Ray functions are not detected by codecov (thus pragma: no cover)"""
    index = partition_indices[0].append(partition_indices[1:])
    index.names = index_name
    return index


class RayIO(BaseIO):

    frame_mgr_cls = None
    frame_partition_cls = None
    query_compiler_cls = None

    # IMPORTANT NOTE
    #
    # Specify these in the child classes to extend the functionality from this class.
    # The tasks must return a very specific set of objects in the correct order to be
    # correct. The following must be returned from these remote tasks:
    # 1.) A number of partitions equal to the `num_partitions` value. If there is not
    #     enough data to fill the number of partitions, returning empty partitions is
    #     okay as well.
    # 2.) The index object if the index is anything but the default type (`RangeIndex`),
    #     otherwise return the length of the object in the remote task and the logic
    #     will build the `RangeIndex` correctly. May of these methods have a `index_col`
    #     parameter that will tell you whether or not to use the default index.

    read_parquet_remote_task = None
    # For reading parquet files in parallel, this task should read based on the `cols`
    # value in the task signature. Each task will read a subset of the columns.
    #
    # Signature: (path, cols, num_splits, kwargs)

    read_csv_remote_task = None
    # For reading CSV files and other text files in parallel, this task should read
    # based on the offsets in the signature (`start` and `stop` are byte offsets).
    # `prefix_id` is the `b""` prefix for reading with a `BytesIO` object and it will
    # also contain encoding information in the string.
    #
    # Signature: (filepath, num_splits, start, stop, kwargs, prefix_id)

    read_hdf_remote_task = None
    # For reading HDF5 files in parallel, this task should read based on the `columns`
    # parameter in the task signature. Each task will read a subset of the columns.
    #
    # Signature: (path_or_buf, columns, num_splits, kwargs)

    read_feather_remote_task = None
    # For reading Feather file format in parallel, this task should read based on the
    # `columns` parameter in the task signature. Each task will read a subset of the
    # columns.
    #
    # Signature: (path, columns, num_splits)

    read_sql_remote_task = None
    # For reading SQL tables in parallel, this task should read a number of rows based
    # on the `sql` string passed to the task. Each task will be given a different LIMIT
    # and OFFSET as a part of the `sql` query string, so the tasks should perform only
    # the logic required to read the SQL query and determine the Index (information
    # above).
    #
    # Signature: (num_splits, sql, con, index_col, kwargs)

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

        if cls.read_parquet_remote_task is None:
            return super(RayIO, cls).read_parquet(path, engine, columns, **kwargs)

        if not columns:
            pf = ParquetFile(path)
            columns = [
                name
                for name in pf.metadata.schema.names
                if not PQ_INDEX_REGEX.match(name)
            ]
        num_partitions = cls.frame_mgr_cls._compute_num_partitions()
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
                cls.read_parquet_remote_task._remote(
                    args=(path, cols, num_splits, kwargs),
                    num_return_vals=num_splits + 1,
                )
                for cols in col_partitions
            ]
        ).T
        remote_partitions = np.array(
            [
                [cls.frame_partition_cls(obj) for obj in row]
                for row in blk_partitions[:-1]
            ]
        )
        index_len = ray.get(blk_partitions[-1][0])
        index = pandas.RangeIndex(index_len)
        new_query_compiler = cls.query_compiler_cls(
            cls.frame_mgr_cls(remote_partitions), index, columns
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
        names = kwargs.get("names", None)
        index_col = kwargs.get("index_col", None)
        if names is None:
            # For the sake of the empty df, we assume no `index_col` to get the correct
            # column names before we build the index. Because we pass `names` in, this
            # step has to happen without removing the `index_col` otherwise it will not
            # be assigned correctly
            kwargs["index_col"] = None
            names = pandas.read_csv(
                file_open(filepath, "rb"), **dict(kwargs, nrows=0, skipfooter=0)
            ).columns
            kwargs["index_col"] = index_col

        empty_pd_df = pandas.read_csv(
            file_open(filepath, "rb"), **dict(kwargs, nrows=0, skipfooter=0)
        )
        column_names = empty_pd_df.columns
        skipfooter = kwargs.get("skipfooter", None)
        skiprows = kwargs.pop("skiprows", None)
        parse_dates = kwargs.pop("parse_dates", False)
        partition_kwargs = dict(
            kwargs,
            header=None,
            names=names,
            skipfooter=0,
            skiprows=None,
            parse_dates=parse_dates,
        )
        with file_open(filepath, "rb") as f:
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
            total_bytes = file_size(f)
            # Max number of partitions available
            num_parts = cls.frame_mgr_cls._compute_num_partitions()
            # This is the number of splits for the columns
            num_splits = min(len(column_names), num_parts)
            # This is the chunksize each partition will read
            chunk_size = max(1, (total_bytes - f.tell()) // num_parts)

            while f.tell() < total_bytes:
                start = f.tell()
                f.seek(chunk_size, os.SEEK_CUR)
                f.readline()  # Read a whole number of lines
                partition_id = cls.read_csv_remote_task._remote(
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
                    [cls.frame_partition_cls(obj) for obj in partition_id[:-1]]
                )
                index_ids.append(partition_id[-1])

        if index_col is None:
            new_index = pandas.RangeIndex(sum(ray.get(index_ids)))
        else:
            new_index_ids = get_index.remote([empty_pd_df.index.name], *index_ids)
            new_index = ray.get(new_index_ids)

        # If parse_dates is present, the column names that we have might not be
        # the same length as the returned column names. If we do need to modify
        # the column names, we remove the old names from the column names and
        # insert the new one at the front of the Index.
        if parse_dates is not None:
            # Check if is list of lists
            if isinstance(parse_dates, list) and isinstance(parse_dates[0], list):
                for group in parse_dates:
                    new_col_name = "_".join(group)
                    column_names = column_names.drop(group).insert(0, new_col_name)
            # Check if it is a dictionary
            elif isinstance(parse_dates, dict):
                for new_col_name, group in parse_dates.items():
                    column_names = column_names.drop(group).insert(0, new_col_name)

        new_query_compiler = cls.query_compiler_cls(
            cls.frame_mgr_cls(np.array(partition_ids)), new_index, column_names
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
        # TODO: Should we try to be smart about how we load files here, or naively default to pandas?
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
        if cls.read_csv_remote_task is None:
            return super(RayIO, cls).read_csv(**kwargs)
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
            if not file_exists(filepath_or_buffer):
                ErrorMessage.default_to_pandas("File path could not be resolved")
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
        if cls.read_hdf_remote_task is None:
            return super(RayIO, cls).read_hdf(path_or_buf, **kwargs)

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

        num_partitions = cls.frame_mgr_cls._compute_num_partitions()
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
                cls.read_hdf_remote_task._remote(
                    args=(path_or_buf, cols, num_splits, kwargs),
                    num_return_vals=num_splits + 1,
                )
                for cols in col_partitions
            ]
        ).T
        remote_partitions = np.array(
            [
                [cls.frame_partition_cls(obj) for obj in row]
                for row in blk_partitions[:-1]
            ]
        )
        index_len = ray.get(blk_partitions[-1][0])
        index = pandas.RangeIndex(index_len)
        new_query_compiler = cls.query_compiler_cls(
            cls.frame_mgr_cls(remote_partitions), index, columns
        )
        return new_query_compiler

    @classmethod
    def read_feather(cls, path, columns=None, use_threads=True):
        """Read a pandas.DataFrame from Feather format.
           Ray DataFrame only supports pyarrow engine for now.

        Args:
            path: The filepath of the feather file.
                  We only support local files for now.
                multi threading is set to True by default
            columns: not supported by pandas api, but can be passed here to read only
                specific columns
            use_threads: Whether or not to use threads when reading

        Notes:
            pyarrow feather is used. Please refer to the documentation here
            https://arrow.apache.org/docs/python/api.html#feather-format
        """
        if cls.read_feather_remote_task is None:
            return super(RayIO, cls).read_feather(
                path, columns=columns, use_threads=use_threads
            )

        if columns is None:
            from pyarrow.feather import FeatherReader

            fr = FeatherReader(path)
            columns = [fr.get_column_name(i) for i in range(fr.num_columns)]

        num_partitions = cls.frame_mgr_cls._compute_num_partitions()
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
                cls.read_feather_remote_task._remote(
                    args=(path, cols, num_splits), num_return_vals=num_splits + 1
                )
                for cols in col_partitions
            ]
        ).T
        remote_partitions = np.array(
            [
                [cls.frame_partition_cls(obj) for obj in row]
                for row in blk_partitions[:-1]
            ]
        )
        index_len = ray.get(blk_partitions[-1][0])
        index = pandas.RangeIndex(index_len)
        new_query_compiler = cls.query_compiler_cls(
            cls.frame_mgr_cls(remote_partitions), index, columns
        )
        return new_query_compiler

    @classmethod
    def to_sql(cls, qc, **kwargs):
        """Write records stored in a DataFrame to a SQL database.
        Args:
            qc: the query compiler of the DF that we want to run to_sql on
            kwargs: parameters for pandas.to_sql(**kwargs)
        """
        # we first insert an empty DF in order to create the full table in the database
        # This also helps to validate the input against pandas
        # we would like to_sql() to complete only when all rows have been inserted into the database
        # since the mapping operation is non-blocking, each partition will return an empty DF
        # so at the end, the blocking operation will be this empty DF to_pandas

        empty_df = qc.head(1).to_pandas().head(0)
        empty_df.to_sql(**kwargs)
        # so each partition will append its respective DF
        kwargs["if_exists"] = "append"
        columns = qc.columns

        def func(df, **kwargs):
            df.columns = columns
            df.to_sql(**kwargs)
            return pandas.DataFrame()

        map_func = qc._prepare_method(func, **kwargs)
        result = qc._map_across_full_axis(1, map_func)
        # blocking operation
        result.to_pandas()

    @classmethod
    def read_sql(cls, sql, con, index_col=None, **kwargs):
        """Reads a SQL query or database table into a DataFrame.
        Args:
            sql: string or SQLAlchemy Selectable (select or text object) SQL query to be
                executed or a table name.
            con: SQLAlchemy connectable (engine/connection) or database string URI or
                DBAPI2 connection (fallback mode)
            index_col: Column(s) to set as index(MultiIndex).
            kwargs: Pass into pandas.read_sql function.
        """
        if cls.read_sql_remote_task is None:
            return super(RayIO, cls).read_sql(sql, con, index_col=index_col, **kwargs)

        row_cnt_query = "SELECT COUNT(*) FROM ({})".format(sql)
        row_cnt = pandas.read_sql(row_cnt_query, con).squeeze()
        cols_names_df = pandas.read_sql(
            "SELECT * FROM ({}) LIMIT 0".format(sql), con, index_col=index_col
        )
        cols_names = cols_names_df.columns
        num_parts = cls.frame_mgr_cls._compute_num_partitions()
        partition_ids = []
        index_ids = []
        limit = math.ceil(row_cnt / num_parts)
        for part in range(num_parts):
            offset = part * limit
            query = "SELECT * FROM ({}) LIMIT {} OFFSET {}".format(sql, limit, offset)
            partition_id = cls.read_sql_remote_task._remote(
                args=(num_parts, query, con, index_col, kwargs),
                num_return_vals=num_parts + 1,
            )
            partition_ids.append(
                [cls.frame_partition_cls(obj) for obj in partition_id[:-1]]
            )
            index_ids.append(partition_id[-1])

        if index_col is None:  # sum all lens returned from partitions
            index_lens = ray.get(index_ids)
            new_index = pandas.RangeIndex(sum(index_lens))
        else:  # concat index returned from partitions
            index_lst = [x for part_index in ray.get(index_ids) for x in part_index]
            new_index = pandas.Index(index_lst).set_names(index_col)

        new_query_compiler = cls.query_compiler_cls(
            cls.frame_mgr_cls(np.array(partition_ids)), new_index, cols_names
        )
        return new_query_compiler
