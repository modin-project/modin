from modin.engines.base.io.file_reader import FileReader
import re
import numpy as np


class TextFileReader(FileReader):
    @classmethod
    def call_deploy(cls, f, chunk_size, num_return_vals, args, quotechar=b'"'):
        args["start"] = f.tell()
        chunk = f.read(chunk_size)
        line = f.readline()  # Ensure we read up to a newline
        # We need to ensure that
        quote_count = (
            re.subn(quotechar, b"", chunk)[1] + re.subn(quotechar, b"", line)[1]
        )
        while quote_count % 2 != 0:
            quote_count += re.subn(quotechar, b"", f.readline())[1]
        # The workers return multiple objects for each part of the file read:
        # - The first n - 2 objects are partitions of data
        # - The n - 1 object is the length of the partition or the index if
        #   `index_col` is specified. We compute the index below.
        # - The nth object is the dtypes of the partition. We combine these to
        #   form the final dtypes below.
        args["end"] = f.tell()
        return cls.deploy(cls.parse, num_return_vals, args)

    @classmethod
    def build_partition(cls, partition_ids, row_lengths, column_widths):
        return np.array(
            [
                [
                    cls.frame_partition_cls(
                        partition_ids[i][j],
                        length=row_lengths[i],
                        width=column_widths[j],
                    )
                    for j in range(len(partition_ids[i]))
                ]
                for i in range(len(partition_ids))
            ]
        )
