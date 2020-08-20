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
from pandas.core.dtypes.common import is_list_like
import numpy as np
import warnings
import csv
import os


class TextFileReader(FileReader):
    @classmethod
    def call_deploy(
        cls,
        f,
        num_return_vals,
        args,
        quotechar=b'"',
        nrows=None,
        chunk_size_bytes=-1,
        rows_behind=None,
        **kwargs,
    ):
        args["start"] = f.tell()
        is_quoting = args.get("quoting", "") != csv.QUOTE_NONE
        outside_quotes = True
        rows_readed = None
        rows_considered = None

        skiprows = args.get("skiprows", None)
        should_handle_skiprows = skiprows is not None and not isinstance(skiprows, int)

        if should_handle_skiprows:
            # slow path, hopefully never getting this way
            if rows_behind is None:
                _, rows_behind = cls.read_nrows(
                    f,
                    quotechar=quotechar,
                    is_quoting=is_quoting,
                    max_bytes=args["start"],
                    from_begin=True,
                    save_position=True,
                )

            args = args.copy()
            args["skiprows"] = cls.handle_skiprows(
                args["skiprows"], rows_behind, **kwargs,
            )
        else:
            rows_considered = rows_readed

        # We want to avoid unnecessary overhead of counting amount
        # of readed rows if we don't need that value
        if not should_handle_skiprows and nrows is None:
            chunk = f.read(chunk_size_bytes)
            line = f.readline()  # Ensure we read up to a newline
            # We need to ensure that one row isn't being split across different partitions

            if is_quoting:
                outside_quotes = not (
                    (chunk.count(quotechar) + line.count(quotechar)) % 2
                )
                while not outside_quotes:
                    line = f.readline()
                    outside_quotes = line.count(quotechar) % 2
                    if not line:
                        break
        else:
            chunk_size_bytes = None if chunk_size_bytes < 0 else chunk_size_bytes
            outside_quotes, rows_readed, rows_considered = cls.read_nrows(
                f,
                nrows,
                quotechar,
                is_quoting,
                max_bytes=chunk_size_bytes,
                skiprows=(args["skiprows"] if should_handle_skiprows else None),
            )

        if is_quoting and not outside_quotes:
            warnings.warn("File has mismatched quotes")

        # The workers return multiple objects for each part of the file read:
        # - The first n - 2 objects are partitions of data
        # - The n - 1 object is the length of the partition or the index if
        #   `index_col` is specified. We compute the index below.
        # - The nth object is the dtypes of the partition. We combine these to
        #   form the final dtypes below.
        args["end"] = f.tell()

        return (
            cls.deploy(cls.parse, num_return_vals, args),
            rows_readed,
            rows_considered,
        )

    @classmethod
    def build_partition(cls, partition_ids, row_lengths, column_widths):
        # breakpoint()
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

    @classmethod
    def read_nrows(
        cls,
        f,
        nrows=None,
        quotechar=b'"',
        is_quoting=True,
        max_bytes=None,
        from_begin=False,
        save_position=False,
        skiprows=None,
    ):
        if nrows is None and max_bytes is None:
            max_bytes = float("inf")

        if nrows is not None and nrows <= 0:
            return True, 0, 0

        # we need this condition to avoid unnecessary checks in `stop_condition`
        # which executes in a huge for loop
        if nrows is not None and max_bytes is None:
            stop_condition = lambda rows_readed: rows_readed >= nrows  # noqa (E731)
        elif nrows is not None and max_bytes is not None:
            stop_condition = (
                lambda rows_readed: f.tell() >= max_bytes or rows_readed >= nrows
            )  # noqa (E731)
        else:
            stop_condition = lambda rows_readed: f.tell() >= max_bytes  # noqa (E731)

        if max_bytes is not None:
            max_bytes = max_bytes + f.tell()

        start_position = f.tell()
        if from_begin:
            f.seek(0, os.SEEK_SET)

        rows_considered = 0
        rows_readed = 0
        outside_quotes = True

        should_handle_skiprows = skiprows is not None and not isinstance(skiprows, int)

        def skiprows_handler_builder(skiprows):
            if callable(skiprows):

                def stepper():
                    row_number = 0
                    while True:
                        yield not skiprows(row_number)
                        row_number += 1

            elif is_list_like(skiprows):

                def stepper():
                    row_number = 0
                    index_to_compare = 0
                    while index_to_compare < len(skiprows):
                        if skiprows[index_to_compare] == row_number:
                            index_to_compare += 1
                            yield 0
                        else:
                            yield 1
                        row_number += 1
                    while True:
                        yield 1

            else:

                def stepper():
                    while True:
                        yield 1

            return stepper()

        if should_handle_skiprows:
            skiprows_handler = skiprows_handler_builder(skiprows)

        for line in f:
            if is_quoting and line.count(quotechar) % 2:
                outside_quotes = not outside_quotes
            if outside_quotes:
                if should_handle_skiprows:
                    rows_considered += next(skiprows_handler)
                else:
                    rows_considered += 1
                rows_readed += 1
                if stop_condition(rows_readed=rows_considered):
                    break
        if not outside_quotes:
            rows_readed += 1

        if save_position:
            f.seek(start_position, os.SEEK_SET)

        return outside_quotes, rows_readed, rows_considered

    @classmethod
    def handle_skiprows(cls, skiprows, chunk_start_row, extra_skiprows=None):
        """
        Desrtiption

        Parameters
        ----------
        skiprows:
        chunk_start_row:
        chunk_end_row:
        extra_skiprows:

        Returns
        -------
        """
        if extra_skiprows is None:
            extra_skiprows = []

        def skiprows_wrapper(n):
            return n in extra_skiprows or skiprows(n + chunk_start_row)

        if callable(skiprows):
            new_skiprows = skiprows_wrapper
        elif is_list_like(skiprows):
            start = np.searchsorted(skiprows, chunk_start_row)
            new_skiprows = np.concatenate(
                [extra_skiprows, skiprows[start:] - chunk_start_row]
            )
            if len(extra_skiprows) > 0:
                new_skiprows = np.sort(new_skiprows)
        else:
            new_skiprows = skiprows

        return new_skiprows

    @classmethod
    def rows_skipper_builder(cls, f, quotechar, is_quoting, skiprows=None):
        _skiprows = skiprows

        def skipper(n, skiprows=None):
            skiprows = skiprows if skiprows is not None else _skiprows
            return cls.read_nrows(
                f,
                quotechar=quotechar,
                is_quoting=is_quoting,
                nrows=n,
                skiprows=skiprows,
            )[1]

        return skipper
