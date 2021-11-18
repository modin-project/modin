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

from collections import OrderedDict
from io import BytesIO
import numpy as np
import pandas
from pandas.core.dtypes.cast import find_common_type
from pandas.core.dtypes.concat import union_categoricals
from pandas.io.common import infer_compression
import warnings

from modin.core.io.file_dispatcher import OpenFile
from modin.core.execution.ray.implementations.cudf_on_ray.partitioning.partition_manager import (
    GPU_MANAGERS,
)
from modin.core.storage_formats.pandas.utils import split_result_of_axis_func_pandas
from modin.error_message import ErrorMessage


def _split_result_for_readers(axis, num_splits, df):  # pragma: no cover
    """Splits the DataFrame read into smaller DataFrames and handles all edge cases.

    Args:
        axis: Which axis to split over.
        num_splits: The number of splits to create.
        df: The DataFrame after it has been read.

    Returns:
        A list of pandas DataFrames.
    """
    splits = split_result_of_axis_func_pandas(axis, num_splits, df)
    if not isinstance(splits, list):
        splits = [splits]
    return splits


def find_common_type_cat(types):
    if all(isinstance(t, pandas.CategoricalDtype) for t in types):
        if all(t.ordered for t in types):
            return pandas.CategoricalDtype(
                np.sort(np.unique([c for t in types for c in t.categories])[0]),
                ordered=True,
            )
        return union_categoricals(
            [pandas.Categorical([], dtype=t) for t in types],
            sort_categories=all(t.ordered for t in types),
        ).dtype
    else:
        return find_common_type(types)


class cuDFParser(object):
    @classmethod
    def get_dtypes(cls, dtypes_ids):
        return (
            pandas.concat(cls.materialize(dtypes_ids), axis=1)
            .apply(lambda row: find_common_type_cat(row.values), axis=1)
            .squeeze(axis=0)
        )

    @classmethod
    def single_worker_read(cls, fname, **kwargs):
        ErrorMessage.default_to_pandas("Parameters provided")
        # Use default args for everything
        pandas_frame = cls.parse(fname, **kwargs)
        if isinstance(pandas_frame, pandas.io.parsers.TextFileReader):
            pd_read = pandas_frame.read
            pandas_frame.read = (
                lambda *args, **kwargs: cls.query_compiler_cls.from_pandas(
                    pd_read(*args, **kwargs), cls.frame_cls
                )
            )
            return pandas_frame
        elif isinstance(pandas_frame, (OrderedDict, dict)):
            return {
                i: cls.query_compiler_cls.from_pandas(frame, cls.frame_cls)
                for i, frame in pandas_frame.items()
            }
        return cls.query_compiler_cls.from_pandas(pandas_frame, cls.frame_cls)

    infer_compression = infer_compression


class cuDFCSVParser(cuDFParser):
    @classmethod
    def parse(cls, fname, **kwargs):
        warnings.filterwarnings("ignore")
        num_splits = kwargs.pop("num_splits", None)
        start = kwargs.pop("start", None)
        end = kwargs.pop("end", None)
        index_col = kwargs.get("index_col", None)
        gpu_selected = kwargs.pop("gpu", 0)

        if start is not None and end is not None:
            put_func = cls.frame_partition_cls.put

            # pop "compression" from kwargs because bio is uncompressed
            with OpenFile(fname, "rb", kwargs.pop("compression", "infer")) as bio:
                if kwargs.get("encoding", None) is not None:
                    header = b"" + bio.readline()
                else:
                    header = b""
                bio.seek(start)
                to_read = header + bio.read(end - start)
            pandas_df = pandas.read_csv(BytesIO(to_read), **kwargs)
        else:
            # This only happens when we are reading with only one worker (Default)
            pandas_df = pandas.read_csv(fname, **kwargs)
            num_splits = (
                1  # force num_splits to be 1 here because we don't want it partitioning
            )
        if index_col is not None:
            index = pandas_df.index
        else:
            index = len(pandas_df)
        partition_dfs = _split_result_for_readers(1, num_splits, pandas_df)
        key = [
            put_func(GPU_MANAGERS[gpu_selected], partition_df)
            for partition_df in partition_dfs
        ]
        return key + [index, pandas_df.dtypes]
