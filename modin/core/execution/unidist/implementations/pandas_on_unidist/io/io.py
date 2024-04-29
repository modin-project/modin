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

"""The module holds the factory which performs I/O using pandas on unidist."""

import io

import numpy as np
import pandas
from pandas.io.common import get_handle, stringify_path

from modin.core.execution.unidist.common import SignalActor, UnidistWrapper
from modin.core.execution.unidist.generic.io import UnidistIO
from modin.core.io import (
    CSVDispatcher,
    ExcelDispatcher,
    FeatherDispatcher,
    FWFDispatcher,
    JSONDispatcher,
    ParquetDispatcher,
    SQLDispatcher,
)
from modin.core.storage_formats.pandas.parsers import (
    PandasCSVParser,
    PandasExcelParser,
    PandasFeatherParser,
    PandasFWFParser,
    PandasJSONParser,
    PandasParquetParser,
    PandasSQLParser,
)
from modin.core.storage_formats.pandas.query_compiler import PandasQueryCompiler
from modin.experimental.core.io import (
    ExperimentalCSVGlobDispatcher,
    ExperimentalCustomTextDispatcher,
    ExperimentalGlobDispatcher,
    ExperimentalSQLDispatcher,
)
from modin.experimental.core.storage_formats.pandas.parsers import (
    ExperimentalCustomTextParser,
    ExperimentalPandasCSVGlobParser,
    ExperimentalPandasJsonParser,
    ExperimentalPandasParquetParser,
    ExperimentalPandasPickleParser,
    ExperimentalPandasXmlParser,
)

from ..dataframe import PandasOnUnidistDataframe
from ..partitioning import PandasOnUnidistDataframePartition


class PandasOnUnidistIO(UnidistIO):
    """Factory providing methods for performing I/O operations using pandas as storage format on unidist as engine."""

    frame_cls = PandasOnUnidistDataframe
    frame_partition_cls = PandasOnUnidistDataframePartition
    query_compiler_cls = PandasQueryCompiler
    build_args = dict(
        frame_partition_cls=PandasOnUnidistDataframePartition,
        query_compiler_cls=PandasQueryCompiler,
        frame_cls=PandasOnUnidistDataframe,
        base_io=UnidistIO,
    )

    def __make_read(*classes, build_args=build_args):
        # used to reduce code duplication
        return type("", (UnidistWrapper, *classes), build_args).read

    def __make_write(*classes, build_args=build_args):
        # used to reduce code duplication
        return type("", (UnidistWrapper, *classes), build_args).write

    read_csv = __make_read(PandasCSVParser, CSVDispatcher)
    read_fwf = __make_read(PandasFWFParser, FWFDispatcher)
    read_json = __make_read(PandasJSONParser, JSONDispatcher)
    read_parquet = __make_read(PandasParquetParser, ParquetDispatcher)
    to_parquet = __make_write(ParquetDispatcher)
    # Blocked on pandas-dev/pandas#12236. It is faster to default to pandas.
    # read_hdf = __make_read(PandasHDFParser, HDFReader)
    read_feather = __make_read(PandasFeatherParser, FeatherDispatcher)
    read_sql = __make_read(PandasSQLParser, SQLDispatcher)
    to_sql = __make_write(SQLDispatcher)
    read_excel = __make_read(PandasExcelParser, ExcelDispatcher)

    # experimental methods that don't exist in pandas
    read_csv_glob = __make_read(
        ExperimentalPandasCSVGlobParser, ExperimentalCSVGlobDispatcher
    )
    read_parquet_glob = __make_read(
        ExperimentalPandasParquetParser, ExperimentalGlobDispatcher
    )
    to_parquet_glob = __make_write(
        ExperimentalGlobDispatcher,
        build_args={**build_args, "base_write": UnidistIO.to_parquet},
    )
    read_json_glob = __make_read(
        ExperimentalPandasJsonParser, ExperimentalGlobDispatcher
    )
    to_json_glob = __make_write(
        ExperimentalGlobDispatcher,
        build_args={**build_args, "base_write": UnidistIO.to_json},
    )
    read_xml_glob = __make_read(ExperimentalPandasXmlParser, ExperimentalGlobDispatcher)
    to_xml_glob = __make_write(
        ExperimentalGlobDispatcher,
        build_args={**build_args, "base_write": UnidistIO.to_xml},
    )
    read_pickle_glob = __make_read(
        ExperimentalPandasPickleParser, ExperimentalGlobDispatcher
    )
    to_pickle_glob = __make_write(
        ExperimentalGlobDispatcher,
        build_args={**build_args, "base_write": UnidistIO.to_pickle},
    )
    read_custom_text = __make_read(
        ExperimentalCustomTextParser, ExperimentalCustomTextDispatcher
    )
    read_sql_distributed = __make_read(
        ExperimentalSQLDispatcher, build_args={**build_args, "base_read": read_sql}
    )

    del __make_read  # to not pollute class namespace
    del __make_write  # to not pollute class namespace

    @staticmethod
    def _to_csv_check_support(kwargs):
        """
        Check if parallel version of ``to_csv`` could be used.

        Parameters
        ----------
        kwargs : dict
            Keyword arguments passed to ``.to_csv()``.

        Returns
        -------
        bool
            Whether parallel version of ``to_csv`` is applicable.
        """
        path_or_buf = kwargs["path_or_buf"]
        compression = kwargs["compression"]
        if not isinstance(path_or_buf, str):
            return False
        # case when the pointer is placed at the beginning of the file.
        if "r" in kwargs["mode"] and "+" in kwargs["mode"]:
            return False
        # encodings with BOM don't support;
        # instead of one mark in result bytes we will have them by the number of partitions
        # so we should fallback in pandas for `utf-16`, `utf-32` with all aliases, in instance
        # (`utf_32_be`, `utf_16_le` and so on)
        if kwargs["encoding"] is not None:
            encoding = kwargs["encoding"].lower()
            if "u" in encoding or "utf" in encoding:
                if "16" in encoding or "32" in encoding:
                    return False
        if compression is None or not compression == "infer":
            return False
        if any((path_or_buf.endswith(ext) for ext in [".gz", ".bz2", ".zip", ".xz"])):
            return False
        return True

    @classmethod
    def to_csv(cls, qc, **kwargs):
        """
        Write records stored in the `qc` to a CSV file.

        Parameters
        ----------
        qc : BaseQueryCompiler
            The query compiler of the Modin dataframe that we want to run ``to_csv`` on.
        **kwargs : dict
            Parameters for ``pandas.to_csv(**kwargs)``.
        """
        kwargs["path_or_buf"] = stringify_path(kwargs["path_or_buf"])
        if not cls._to_csv_check_support(kwargs):
            return UnidistIO.to_csv(qc, **kwargs)

        signals = SignalActor.remote(len(qc._modin_frame._partitions) + 1)

        def func(df, **kw):  # pragma: no cover
            """
            Dump a chunk of rows as csv, then save them to target maintaining order.

            Parameters
            ----------
            df : pandas.DataFrame
                A chunk of rows to write to a CSV file.
            **kw : dict
                Arguments to pass to ``pandas.to_csv(**kw)`` plus an extra argument
                `partition_idx` serving as chunk index to maintain rows order.
            """
            partition_idx = kw["partition_idx"]
            # the copy is made to not implicitly change the input parameters;
            # to write to an intermediate buffer, we need to change `path_or_buf` in kwargs
            csv_kwargs = kwargs.copy()
            if partition_idx != 0:
                # we need to create a new file only for first recording
                # all the rest should be recorded in appending mode
                if "w" in csv_kwargs["mode"]:
                    csv_kwargs["mode"] = csv_kwargs["mode"].replace("w", "a")
                # It is enough to write the header for the first partition
                csv_kwargs["header"] = False

            # for parallelization purposes, each partition is written to an intermediate buffer
            path_or_buf = csv_kwargs["path_or_buf"]
            is_binary = "b" in csv_kwargs["mode"]
            csv_kwargs["path_or_buf"] = io.BytesIO() if is_binary else io.StringIO()
            storage_options = csv_kwargs.pop("storage_options", None)
            df.to_csv(**csv_kwargs)
            csv_kwargs.update({"storage_options": storage_options})
            content = csv_kwargs["path_or_buf"].getvalue()
            csv_kwargs["path_or_buf"].close()

            # each process waits for its turn to write to a file
            UnidistWrapper.materialize(signals.wait.remote(partition_idx))

            # preparing to write data from the buffer to a file
            with get_handle(
                path_or_buf,
                # in case when using URL in implicit text mode
                # pandas try to open `path_or_buf` in binary mode
                csv_kwargs["mode"] if is_binary else csv_kwargs["mode"] + "t",
                encoding=kwargs["encoding"],
                errors=kwargs["errors"],
                compression=kwargs["compression"],
                storage_options=kwargs.get("storage_options", None),
                is_text=not is_binary,
            ) as handles:
                handles.handle.write(content)

            # signal that the next process can start writing to the file
            UnidistWrapper.materialize(signals.send.remote(partition_idx + 1))
            # used for synchronization purposes
            return pandas.DataFrame()

        # signaling that the partition with id==0 can be written to the file
        UnidistWrapper.materialize(signals.send.remote(0))
        # Ensure that the metadata is syncrhonized
        qc._modin_frame._propagate_index_objs(axis=None)
        result = qc._modin_frame._partition_mgr_cls.map_axis_partitions(
            axis=1,
            partitions=qc._modin_frame._partitions,
            map_func=func,
            keep_partitioning=True,
            lengths=None,
            enumerate_partitions=True,
            max_retries=0,
        )
        # pending completion
        UnidistWrapper.materialize(
            [part.list_of_blocks[0] for row in result for part in row]
        )

    @classmethod
    def from_map(cls, func, iterable, *args, **kwargs):
        """
        Create a Modin `query_compiler` from a map function.

        This method will construct a Modin `query_compiler` split by row partitions.
        The number of row partitions matches the number of elements in the iterable object.

        Parameters
        ----------
        func : callable
            Function to map across the iterable object.
        iterable : Iterable
            An iterable object.
        *args : tuple
            Positional arguments to pass in `func`.
        **kwargs : dict
            Keyword arguments to pass in `func`.

        Returns
        -------
        BaseQueryCompiler
            QueryCompiler containing data returned by map function.
        """
        func = cls.frame_cls._partition_mgr_cls.preprocess_func(func)
        partitions = np.array(
            [
                [
                    cls.frame_partition_cls(
                        UnidistWrapper.deploy(
                            func,
                            f_args=(obj,) + args,
                            f_kwargs=kwargs,
                            return_pandas_df=True,
                        )
                    )
                ]
                for obj in iterable
            ]
        )
        return cls.query_compiler_cls(cls.frame_cls(partitions))
