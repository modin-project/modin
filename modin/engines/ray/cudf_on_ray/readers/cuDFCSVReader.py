
from modin.engines.base.io.text.text_file_reader import TextFileReader
from modin.data_management.utils import compute_chunksize
from pandas.io.parsers import _validate_usecols_arg
import pandas
import sys
import numpy as np

class cuDFCSVReader(TextFileReader):
    @classmethod
    def _read(cls, filepath_or_buffer, **kwargs):
        if isinstance(filepath_or_buffer, str):
            if not cls.file_exists(filepath_or_buffer):
                return cls.single_worker_read(filepath_or_buffer, **kwargs)
            filepath_or_buffer = cls.get_path(filepath_or_buffer)
        elif not cls.pathlib_or_pypath(filepath_or_buffer):
            return cls.single_worker_read(filepath_or_buffer, **kwargs)
        compression_type = cls.infer_compression(
            filepath_or_buffer, kwargs.get("compression")
        )
        if compression_type is not None:
            if (
                compression_type == "gzip"
                or compression_type == "bz2"
                or compression_type == "xz"
            ):
                kwargs["compression"] = compression_type
            elif (
                compression_type == "zip"
                and sys.version_info[0] == 3
                and sys.version_info[1] >= 7
            ):
                # need python3.7 to .seek and .tell ZipExtFile
                kwargs["compression"] = compression_type
            else:
                return cls.single_worker_read(filepath_or_buffer, **kwargs)

        chunksize = kwargs.get("chunksize")
        if chunksize is not None:
            return cls.single_worker_read(filepath_or_buffer, **kwargs)

        skiprows = kwargs.get("skiprows")
        if skiprows is not None and not isinstance(skiprows, int):
            return cls.single_worker_read(filepath_or_buffer, **kwargs)
        # TODO: replace this by reading lines from file.
        if kwargs.get("nrows") is not None:
            return cls.single_worker_read(filepath_or_buffer, **kwargs)
        names = kwargs.get("names", None)
        index_col = kwargs.get("index_col", None)
        if names is None:
            # For the sake of the empty df, we assume no `index_col` to get the correct
            # column names before we build the index. Because we pass `names` in, this
            # step has to happen without removing the `index_col` otherwise it will not
            # be assigned correctly
            names = pandas.read_csv(
                filepath_or_buffer,
                **dict(kwargs, usecols=None, nrows=0, skipfooter=0, index_col=None),
            ).columns
        empty_pd_df = pandas.read_csv(
            filepath_or_buffer, **dict(kwargs, nrows=0, skipfooter=0)
        )
        column_names = empty_pd_df.columns
        skipfooter = kwargs.get("skipfooter", None)
        skiprows = kwargs.pop("skiprows", None)
        usecols = kwargs.get("usecols", None)
        usecols_md = _validate_usecols_arg(usecols)
        if usecols is not None and usecols_md[1] != "integer":
            del kwargs["usecols"]
            all_cols = pandas.read_csv(
                cls.file_open(filepath_or_buffer, "rb"),
                **dict(kwargs, nrows=0, skipfooter=0),
            ).columns
            usecols = all_cols.get_indexer_for(list(usecols_md[0]))
        parse_dates = kwargs.pop("parse_dates", False)
        partition_kwargs = dict(
            kwargs,
            header=None,
            names=names,
            skipfooter=0,
            skiprows=None,
            parse_dates=parse_dates,
            usecols=usecols,
        )
        encoding = kwargs.get("encoding", None)
        quotechar = kwargs.get("quotechar", '"').encode(
            encoding if encoding is not None else "UTF-8"
        )
        with cls.file_open(filepath_or_buffer, "rb", compression_type) as f:
            # Skip the header since we already have the header information and skip the
            # rows we are told to skip.
            if isinstance(skiprows, int) or skiprows is None:
                if skiprows is None:
                    skiprows = 0
                header = kwargs.get("header", "infer")
                if header == "infer" and kwargs.get("names", None) is None:
                    skiprows += 1
                elif isinstance(header, int):
                    skiprows += header + 1
                elif hasattr(header, "__iter__") and not isinstance(header, str):
                    skiprows += max(header) + 1
                for _ in range(skiprows):
                    f.readline()
            if kwargs.get("encoding", None) is not None:
                partition_kwargs["skiprows"] = 1
            # Launch tasks to read partitions
            index_ids = []
            dtypes_ids = []
            total_bytes = cls.file_size(f)

            # Max number of partitions available
            
            num_row_partitions = cls.frame_partition_mgr_cls._compute_num_row_partitions()
            num_col_partitions = cls.frame_partition_mgr_cls._compute_num_col_partitions()
            
            gpu_manager = cls.frame_partition_mgr_cls._get_gpu_managers()

            # This is the number of splits for the columns
            # num_splits = min(len(column_names), num_partitions)
            # This is the chunksize each partition will read
            row_chunksize = max(1, (total_bytes - f.tell()) // num_row_partitions)

            # Metadata
            column_chunksize = compute_chunksize(empty_pd_df, num_col_partitions, axis=1)
            if column_chunksize > len(column_names):
                column_widths = [len(column_names)]
                # This prevents us from unnecessarily serializing a bunch of empty
                # objects.
                num_col_partitions = 1
            else:
                column_widths = [
                    column_chunksize
                    if len(column_names) > (column_chunksize * (i + 1))
                    else 0
                    if len(column_names) < (column_chunksize * i)
                    else len(column_names) - (column_chunksize * i)
                    for i in range(num_col_partitions)
                ]

            gpu_idx = 0
            keys = []

            # num_splits = 1
            num_rows, num_cols = num_row_partitions, num_col_partitions

            while f.tell() < total_bytes:
                assigned_gpus = []
                for i in range(num_col_partitions):
                    assigned_gpus.append(gpu_manager[gpu_idx])
                    gpu_idx += 1
                args = {
                    "fname": filepath_or_buffer,
                    "num_splits": num_col_partitions,
                    "assigned_gpus": assigned_gpus,
                    **partition_kwargs,
                }
                partition_ids = cls.call_deploy(
                    f, row_chunksize, num_col_partitions + 2, args, quotechar=quotechar
                )
                keys.append(partition_ids[:-2])
                index_ids.append(partition_ids[-2])
                dtypes_ids.append(partition_ids[-1])


        ## Compute the index based on a sum of the lengths of each partition (by default)
        ## or based on the column(s) that were requested.
        if index_col is None:
            row_lengths = cls.materialize(index_ids)
            new_index = pandas.RangeIndex(sum(row_lengths))
            # pandas has a really weird edge case here.
            if kwargs.get("names", None) is not None and skiprows > 1:
                new_index = pandas.RangeIndex(
                    skiprows - 1, new_index.stop + skiprows - 1
                )
        else:
            index_objs = cls.materialize(index_ids)
            row_lengths = [len(o) for o in index_objs]
            new_index = index_objs[0].append(index_objs[1:])
            new_index.name = empty_pd_df.index.name

        ### Compute dtypes by getting collecting and combining all of the partitions. The
        ### reported dtypes from differing rows can be different based on the inference in
        ### the limited data seen by each worker. We use pandas to compute the exact dtype
        ### over the whole column for each column. The index is set below.
        dtypes = cls.get_dtypes(dtypes_ids)

        keys = list(np.array(keys).flatten())
        keys = cls.materialize(keys)
        partition_ids = cls.build_partition(zip(gpu_manager, keys)).reshape((num_rows, num_cols))

        new_frame = cls.frame_cls(
            partition_ids,
            new_index,
            column_names,
            row_lengths,
            column_widths,
            dtypes=dtypes,
        )
        new_query_compiler = cls.query_compiler_cls(new_frame)

        return new_query_compiler

    @classmethod
    def build_partition(cls, gpu_managers_and_keys):
        return np.array(
            [
                [cls.frame_partition_cls(gpu_manager, key)]
                for gpu_manager, key in gpu_managers_and_keys
            ], dtype=object
        )
