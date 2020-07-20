# Licensed to Modin Development Team under one or more contributor license agreements.
# See the NOTICE file distributed with this work for additional information regarding
# copyright ownership.  The Modin Development Team licenses this file to you under the
# Apache License, Version 2.0 (the "License"); you may not use this file except in
# compliance with the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under
# the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.

from modin.engines.base.io.file_reader import FileReader
import re
import numpy as np
import warnings
import csv


class TextFileReader(FileReader):
    @classmethod
    def call_deploy(cls, f, chunk_size, num_return_vals, args, quotechar=b'"'):
        args["start"] = f.tell()
        chunk = f.read(chunk_size)
        line = f.readline()  # Ensure we read up to a newline
        # We need to ensure that one row isn't being split across different partitions

        if args.get("quoting", "") != csv.QUOTE_NONE:
            quote_count = (
                re.subn(quotechar, b"", chunk)[1] + re.subn(quotechar, b"", line)[1]
            )
            while quote_count % 2 != 0:
                line = f.readline()
                quote_count += re.subn(quotechar, b"", line)[1]
                if not line:
                    break

            if quote_count % 2 != 0:
                warnings.warn("File has mismatched quotes")

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

    @classmethod
    def pathlib_or_pypath(cls, filepath_or_buffer):
        try:
            import py

            if isinstance(filepath_or_buffer, py.path.local):
                return True
        except ImportError:  # pragma: no cover
            pass
        try:
            import pathlib

            if isinstance(filepath_or_buffer, pathlib.Path):
                return True
        except ImportError:  # pragma: no cover
            pass
        return False
