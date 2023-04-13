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

"""
Module houses experimental IO classes and parser functions needed for these classes.

Any function or class can be considered experimental API if it is not strictly replicating existent
Query Compiler API, even if it is only extending the API.
"""

import warnings

import pandas

from modin.core.storage_formats.pandas.parsers import (
    PandasCSVGlobParser,
    ExperimentalPandasPickleParser,
    ExperimentalCustomTextParser,
)
from modin.core.storage_formats.pandas.query_compiler import PandasQueryCompiler
from modin.core.execution.dask.implementations.pandas_on_dask.io import PandasOnDaskIO
from modin.core.io import (
    CSVGlobDispatcher,
    ExperimentalPickleDispatcher,
    ExperimentalCustomTextDispatcher,
)
from modin.experimental.core.io import ExperimentalSQLDispatcher

from modin.core.execution.dask.implementations.pandas_on_dask.dataframe import (
    PandasOnDaskDataframe,
)
from modin.core.execution.dask.implementations.pandas_on_dask.partitioning import (
    PandasOnDaskDataframePartition,
)
from modin.core.execution.dask.common import DaskWrapper


class ExperimentalPandasOnDaskIO(PandasOnDaskIO):
    """
    Class for handling experimental IO functionality with pandas storage format and Dask engine.

    ``ExperimentalPandasOnDaskIO`` inherits some util functions and unmodified IO functions
    from ``PandasOnDaskIO`` class.
    """

    build_args = dict(
        frame_partition_cls=PandasOnDaskDataframePartition,
        query_compiler_cls=PandasQueryCompiler,
        frame_cls=PandasOnDaskDataframe,
        base_io=PandasOnDaskIO,
    )

    def __make_read(*classes, build_args=build_args):  # noqa: GL08
        # used to reduce code duplication
        return type("", (DaskWrapper, *classes), build_args).read

    read_csv_glob = __make_read(PandasCSVGlobParser, CSVGlobDispatcher)

    read_pickle_distributed = __make_read(
        ExperimentalPandasPickleParser, ExperimentalPickleDispatcher
    )

    read_custom_text = __make_read(
        ExperimentalCustomTextParser, ExperimentalCustomTextDispatcher
    )

    read_sql = __make_read(ExperimentalSQLDispatcher)

    del __make_read  # to not pollute class namespace

    @classmethod
    def to_pickle_distributed(cls, qc, **kwargs):
        """
        When `*` in the filename all partitions are written to their own separate file.

        The filenames is determined as follows:
        - if `*` in the filename then it will be replaced by the increasing sequence 0, 1, 2, …
        - if `*` is not the filename, then will be used default implementation.

        Examples #1: 4 partitions and input filename="partition*.pkl.gz", then filenames will be:
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
            return PandasOnDaskIO.to_pickle(qc, **kwargs)

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
        DaskWrapper.materialize(
            [part.list_of_blocks[0] for row in result._partitions for part in row]
        )
