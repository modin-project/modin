from modin.engines.base.io.text.text_file_reader import TextFileReader
from modin.data_management.utils import compute_chunksize
from io import BytesIO
import pandas
import numpy as np


class JSONReader(TextFileReader):
    @classmethod
    def read(cls, path_or_buf, **kwargs):
        path_or_buf = cls.get_path(path_or_buf)
        if not kwargs.get("lines", False):
            return cls.single_worker_read(path_or_buf, **kwargs)
        columns = pandas.read_json(
            BytesIO(b"" + open(path_or_buf, "rb").readline()), lines=True
        ).columns
        kwargs["columns"] = columns
        empty_pd_df = pandas.DataFrame(columns=columns)

        with cls.file_open(path_or_buf, "rb", kwargs.get("compression", "infer")) as f:
            total_bytes = cls.file_size(f)
            from modin.pandas import DEFAULT_NPARTITIONS

            num_partitions = DEFAULT_NPARTITIONS
            num_splits = min(len(columns), num_partitions)
            chunk_size = max(1, (total_bytes - f.tell()) // num_partitions)

            partition_ids = []
            index_ids = []
            dtypes_ids = []

            column_chunksize = compute_chunksize(empty_pd_df, num_splits, axis=1)
            if column_chunksize > len(columns):
                column_widths = [len(columns)]
                num_splits = 1
            else:
                column_widths = [
                    column_chunksize
                    if i != num_splits - 1
                    else len(columns) - (column_chunksize * (num_splits - 1))
                    for i in range(num_splits)
                ]

            while f.tell() < total_bytes:
                start = f.tell()
                args = {"fname": path_or_buf, "num_splits": num_splits, "start": start}
                args.update(kwargs)
                partition_id = cls.call_deploy(f, chunk_size, num_splits + 3, args)
                partition_ids.append(partition_id[:-3])
                index_ids.append(partition_id[-3])
                dtypes_ids.append(partition_id[-2])

        # partition_id[-1] contains the columns for each partition, which will be useful
        # for implementing when `lines=False`.
        row_lengths = cls.materialize(index_ids)
        new_index = pandas.RangeIndex(sum(row_lengths))

        dtypes = cls.get_dtypes(dtypes_ids)
        partition_ids = cls.build_partition(partition_ids, row_lengths, column_widths)

        if isinstance(dtypes, pandas.Series):
            dtypes.index = columns
        else:
            dtypes = pandas.Series(dtypes, index=columns)

        new_frame = cls.frame_cls(
            np.array(partition_ids),
            new_index,
            columns,
            row_lengths,
            column_widths,
            dtypes=dtypes,
        )
        new_frame._apply_index_objs(axis=0)
        return cls.query_compiler_cls(new_frame)
