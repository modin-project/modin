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

import pandas
import warnings

from modin.core.storage_formats.pandas.parsers import (
    PandasCSVGlobParser,
    PandasPickleExperimentalParser,
    CustomTextExperimentalParser,
)
from modin.core.storage_formats.pandas.query_compiler import PandasQueryCompiler
from modin.core.execution.unidist.implementations.pandas_on_unidist.io import (
    PandasOnUnidistIO,
)
from modin.core.io import (
    CSVGlobDispatcher,
    PickleExperimentalDispatcher,
    CustomTextExperimentalDispatcher,
)
from modin.experimental.core.io import SQLExperimentalDispatcher
from modin.core.execution.unidist.implementations.pandas_on_unidist.dataframe import (
    PandasOnUnidistDataframe,
)
from modin.core.execution.unidist.implementations.pandas_on_unidist.partitioning import (
    PandasOnUnidistDataframePartition,
)
from modin.core.execution.unidist.common import UnidistWrapper


class ExperimentalPandasOnUnidistIO(PandasOnUnidistIO):
    """
    Class for handling experimental IO functionality with pandas storage format and unidist engine.

    ``ExperimentalPandasOnUnidistIO`` inherits some util functions and unmodified IO functions
    from ``PandasOnUnidistIO`` class.
    """

    build_args = dict(
        frame_partition_cls=PandasOnUnidistDataframePartition,
        query_compiler_cls=PandasQueryCompiler,
        frame_cls=PandasOnUnidistDataframe,
        base_io=PandasOnUnidistIO,
    )

    @classmethod
    def make_read(cls, *classes):  # noqa: GL08
        # used to reduce code duplication
        return type("", (UnidistWrapper, *classes), cls.build_args)._read

    read_csv_glob = make_read(PandasCSVGlobParser, CSVGlobDispatcher)

    read_pickle_distributed = make_read(
        PandasPickleExperimentalParser, PickleExperimentalDispatcher
    )

    read_custom_text = make_read(
        CustomTextExperimentalParser, CustomTextExperimentalDispatcher
    )

    read_sql = make_read(SQLExperimentalDispatcher)

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
            return PandasOnUnidistIO.to_pickle(qc, **kwargs)

        def func(df, **kw):
            idx = str(kw["partition_idx"])
            kwargs["path"] = kwargs.pop("filepath_or_buffer").replace("*", idx)
            df.to_pickle(**kwargs)
            return pandas.DataFrame()

        result = qc._modin_frame.broadcast_apply_full_axis(
            1, func, other=None, new_index=[], new_columns=[], enumerate_partitions=True
        )
        # pending completion
        result.to_pandas()
