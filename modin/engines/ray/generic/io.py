import pandas

import os
import re
import numpy as np
import math
import warnings

from modin.error_message import ErrorMessage
from modin.engines.base.io import BaseIO
from modin.data_management.utils import compute_chunksize
from modin import __execution_engine__

if __execution_engine__ == "Ray":
    import ray

PQ_INDEX_REGEX = re.compile("__index_level_\d+__")  # noqa W605


class RayIO(BaseIO):

    frame_partition_cls = None
    query_compiler_cls = None
    frame_cls = None

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

    read_json_remote_task = None
    # For reading JSON files and other text files in parallel, this task should read
    # based on the offsets in the signature (`start` and `stop` are byte offsets).
    #
    # Signature: (filepath, num_splits, start, stop, kwargs)

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

        from pyarrow.parquet import ParquetFile, ParquetDataset

        if cls.read_parquet_remote_task is None:
            return super(RayIO, cls).read_parquet(path, engine, columns, **kwargs)

        file_path = path
        if os.path.isdir(path):
            directory = True
            partitioned_columns = set()
            # We do a tree walk of the path directory because partitioned
            # parquet directories have a unique column at each directory level.
            # Thus, we can use os.walk(), which does a dfs search, to walk
            # through the different columns that the data is partitioned on
            for (root, dir_names, files) in os.walk(path):
                if dir_names:
                    partitioned_columns.add(dir_names[0].split("=")[0])
                if files:
                    # Metadata files, git files, .DSStore
                    if files[0][0] == ".":
                        continue
                    file_path = os.path.join(root, files[0])
                    break
            partitioned_columns = list(partitioned_columns)
        else:
            directory = False

        if not columns:
            if directory:
                # Path of the sample file that we will read to get the remaining
                # columns.
                from pyarrow import ArrowIOError

                try:
                    pd = ParquetDataset(file_path)
                except ArrowIOError:
                    pd = ParquetDataset(path)
                column_names = pd.schema.names
            else:
                pf = ParquetFile(path)
                column_names = pf.metadata.schema.names
            columns = [name for name in column_names if not PQ_INDEX_REGEX.match(name)]

        # Cannot read in parquet file by only reading in the partitioned column.
        # Thus, we have to remove the partition columns from the columns to
        # ensure that when we do the math for the blocks, the partition column
        # will be read in along with a non partition column.
        if columns and directory and any(col in partitioned_columns for col in columns):
            columns = [col for col in columns if col not in partitioned_columns]
            # If all of the columns wanted are partition columns, return an
            # empty dataframe with the desired columns.
            if len(columns) == 0:
                return cls.from_pandas(pandas.DataFrame(columns=partitioned_columns))

        from modin.pandas import DEFAULT_NPARTITIONS

        num_partitions = DEFAULT_NPARTITIONS
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
        column_widths = [len(c) for c in col_partitions]
        # Each item in this list will be a list of columns of original df
        # partitioned to smaller pieces along rows.
        # We need to transpose the oids array to fit our schema.
        # TODO (williamma12): This part can be parallelized even more if we
        # separate the partitioned parquet file code path from the default one.
        # The workers return multiple objects for each part of the file read:
        # - The first n - 2 objects are partitions of data
        # - The n - 1 object is the length of the partition.
        # - The nth object is the dtypes of the partition. We combine these to
        #   form the final dtypes below.
        blk_partitions = np.array(
            [
                cls.read_parquet_remote_task._remote(
                    args=(path, cols + partitioned_columns, num_splits, kwargs),
                    num_return_vals=num_splits + 2,
                )
                if directory and cols == col_partitions[len(col_partitions) - 1]
                else cls.read_parquet_remote_task._remote(
                    args=(path, cols, num_splits, kwargs),
                    num_return_vals=num_splits + 2,
                )
                for cols in col_partitions
            ]
        ).T
        # Metadata
        index_len = ray.get(blk_partitions[-2][0])
        index = pandas.RangeIndex(index_len)
        index_chunksize = compute_chunksize(
            pandas.DataFrame(index=index), num_splits, axis=0
        )
        if index_chunksize > index_len:
            row_lengths = [index_len] + [0 for _ in range(num_splits - 1)]
        else:
            row_lengths = [
                index_chunksize
                if i != num_splits - 1
                else index_len - (index_chunksize * (num_splits - 1))
                for i in range(num_splits)
            ]
        remote_partitions = np.array(
            [
                [
                    cls.frame_partition_cls(
                        blk_partitions[i][j],
                        length=row_lengths[i],
                        width=column_widths[j],
                    )
                    for j in range(len(blk_partitions[i]))
                ]
                for i in range(len(blk_partitions[:-2]))
            ]
        )
        # Compute dtypes concatenating the results from each of the columns splits
        # determined above. This creates a pandas Series that contains a dtype for every
        # column.
        dtypes_ids = list(blk_partitions[-1])
        dtypes = pandas.concat(ray.get(dtypes_ids), axis=0)
        if directory:
            if len(partitioned_columns) > 0:
                columns += partitioned_columns
                # We will have to recompute this because the lengths do not contain the
                # partitioned columns lengths.
                column_widths = None
        dtypes.index = columns
        new_query_compiler = cls.query_compiler_cls(
            cls.frame_cls(
                remote_partitions,
                index,
                columns,
                row_lengths,
                column_widths,
                dtypes=dtypes,
            )
        )

        return new_query_compiler

    @classmethod
    def read_json(
        cls,
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
        kwargs = {
            "path_or_buf": path_or_buf,
            "orient": orient,
            "typ": typ,
            "dtype": dtype,
            "convert_axes": convert_axes,
            "convert_dates": convert_dates,
            "keep_default_dates": keep_default_dates,
            "numpy": numpy,
            "precise_float": precise_float,
            "date_unit": date_unit,
            "encoding": encoding,
            "lines": lines,
            "chunksize": chunksize,
            "compression": compression,
        }
        return super(RayIO, cls).read_json(**kwargs)

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
            start = kwargs.pop("start", None)
            stop = kwargs.pop("stop", None)
            empty_pd_df = pandas.read_hdf(path_or_buf, start=0, stop=0, **kwargs)
            kwargs["start"] = start
            kwargs["stop"] = stop
            columns = empty_pd_df.columns

        from modin.pandas import DEFAULT_NPARTITIONS

        num_partitions = DEFAULT_NPARTITIONS
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
            cls.frame_cls(remote_partitions, index, columns)
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

        from modin.pandas import DEFAULT_NPARTITIONS

        num_partitions = DEFAULT_NPARTITIONS
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
            cls.frame_cls(remote_partitions, index, columns)
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

        def func(df):
            df.columns = columns
            df.to_sql(**kwargs)
            return pandas.DataFrame()

        result = qc._modin_frame._fold_reduce(1, func)
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

        import sqlalchemy as sa

        # In the case that we are given a SQLAlchemy Connection or Engine, the objects
        # are not pickleable. We have to convert it to the URL string and connect from
        # each of the workers.
        if isinstance(con, (sa.engine.Engine, sa.engine.Connection)):
            warnings.warn(
                "To use parallel implementation of `read_sql`, pass the "
                "connection string instead of {}.".format(type(con))
            )
            return super(RayIO, cls).read_sql(sql, con, index_col=index_col, **kwargs)
        row_cnt_query = "SELECT COUNT(*) FROM ({}) as foo".format(sql)
        row_cnt = pandas.read_sql(row_cnt_query, con).squeeze()
        cols_names_df = pandas.read_sql(
            "SELECT * FROM ({}) as foo LIMIT 0".format(sql), con, index_col=index_col
        )
        cols_names = cols_names_df.columns
        from modin.pandas import DEFAULT_NPARTITIONS

        num_partitions = DEFAULT_NPARTITIONS
        partition_ids = []
        index_ids = []
        limit = math.ceil(row_cnt / num_partitions)
        for part in range(num_partitions):
            offset = part * limit
            query = "SELECT * FROM ({}) as foo LIMIT {} OFFSET {}".format(
                sql, limit, offset
            )
            partition_id = cls.read_sql_remote_task._remote(
                args=(num_partitions, query, con, index_col, kwargs),
                num_return_vals=num_partitions + 1,
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

        new_frame = cls.frame_cls(np.array(partition_ids), new_index, cols_names)
        new_frame._apply_index_objs(axis=0)
        return cls.query_compiler_cls(new_frame)
