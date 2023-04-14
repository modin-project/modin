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

"""Module houses ``ExperimentalPickleDispatcher`` class that is used for reading `.pkl` files."""

import glob
import warnings

import pandas

from modin.core.io.file_dispatcher import FileDispatcher
from modin.config import NPartitions
from modin.core.storage_formats.pandas.query_compiler import PandasQueryCompiler


class ExperimentalPickleDispatcher(FileDispatcher):
    """Class handles utils for reading pickle files."""

    @classmethod
    def _read(cls, filepath_or_buffer, **kwargs):
        """
        Read data from `filepath_or_buffer` according to `kwargs` parameters.

        Parameters
        ----------
        filepath_or_buffer : str, path object or file-like object
            `filepath_or_buffer` parameter of `read_pickle` function.
        **kwargs : dict
            Parameters of `read_pickle` function.

        Returns
        -------
        new_query_compiler : BaseQueryCompiler
            Query compiler with imported data for further processing.

        Notes
        -----
        In experimental mode, we can use `*` in the filename.

        The number of partitions is equal to the number of input files.
        """
        if not (isinstance(filepath_or_buffer, str) and "*" in filepath_or_buffer):
            return cls.single_worker_read(
                filepath_or_buffer,
                single_worker_read=True,
                reason="Buffers and single files are not supported",
                **kwargs,
            )
        filepath_or_buffer = sorted(glob.glob(filepath_or_buffer))

        if len(filepath_or_buffer) == 0:
            raise ValueError(
                f"There are no files matching the pattern: {filepath_or_buffer}"
            )

        partition_ids = [None] * len(filepath_or_buffer)
        lengths_ids = [None] * len(filepath_or_buffer)
        widths_ids = [None] * len(filepath_or_buffer)

        if len(filepath_or_buffer) != NPartitions.get():
            # do we need to do a repartitioning?
            warnings.warn("can be inefficient partitioning")

        for idx, file_name in enumerate(filepath_or_buffer):
            *partition_ids[idx], lengths_ids[idx], widths_ids[idx] = cls.deploy(
                func=cls.parse,
                f_kwargs={
                    "fname": file_name,
                    **kwargs,
                },
                num_returns=3,
            )
        lengths = cls.materialize(lengths_ids)
        widths = cls.materialize(widths_ids)

        # while num_splits is 1, need only one value
        partition_ids = cls.build_partition(partition_ids, lengths, [widths[0]])

        new_index, _ = cls.frame_cls._partition_mgr_cls.get_indices(0, partition_ids)
        new_columns, _ = cls.frame_cls._partition_mgr_cls.get_indices(1, partition_ids)

        return cls.query_compiler_cls(
            cls.frame_cls(partition_ids, new_index, new_columns)
        )

    @classmethod
    def write(cls, qc, **kwargs):
        """
        When `*` is in the filename, all partitions are written to their own separate file.

        The filenames is determined as follows:
        - if `*` is in the filename, then it will be replaced by the ascending sequence 0, 1, 2, …
        - if `*` is not in the filename, then the default implementation will be used.

        Example: 4 partitions and input filename="partition*.pkl.gz", then filenames will be:
        `partition0.pkl.gz`, `partition1.pkl.gz`, `partition2.pkl.gz`, `partition3.pkl.gz`.

        Parameters
        ----------
        qc : BaseQueryCompiler
            The query compiler of the Modin dataframe that we want
            to run ``to_pickle_distributed`` on.
        **kwargs : dict
            Parameters for ``pandas.to_pickle(**kwargs)``.
        """
        if not (
            isinstance(kwargs["filepath_or_buffer"], str)
            and "*" in kwargs["filepath_or_buffer"]
        ) or not isinstance(qc, PandasQueryCompiler):
            warnings.warn("Defaulting to Modin core implementation")
            cls.base_io.to_pickle(qc, **kwargs)
            return

        def func(df, **kw):  # pragma: no cover
            idx = str(kw["partition_idx"])
            # dask doesn't make a copy of kwargs on serialization;
            # so take a copy ourselves, otherwise the error is:
            #  kwargs["path"] = kwargs.pop("filepath_or_buffer").replace("*", idx)
            #  KeyError: 'filepath_or_buffer'
            dask_kwargs = dict(kwargs)
            dask_kwargs["path"] = dask_kwargs.pop("filepath_or_buffer").replace(
                "*", idx
            )
            df.to_pickle(**dask_kwargs)
            return pandas.DataFrame()

        result = qc._modin_frame.apply_full_axis(
            1, func, new_index=[], new_columns=[], enumerate_partitions=True
        )
        cls.materialize(
            [part.list_of_blocks[0] for row in result._partitions for part in row]
        )
