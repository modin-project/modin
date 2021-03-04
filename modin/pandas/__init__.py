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

import pandas

__pandas_version__ = "1.2.3"

if pandas.__version__ != __pandas_version__:
    import warnings

    warnings.warn(
        "The pandas version installed {} does not match the supported pandas version in"
        " Modin {}. This may cause undesired side effects!".format(
            pandas.__version__, __pandas_version__
        )
    )

from pandas import (
    eval,
    cut,
    factorize,
    test,
    qcut,
    date_range,
    period_range,
    Index,
    MultiIndex,
    CategoricalIndex,
    bdate_range,
    DatetimeIndex,
    Timedelta,
    Timestamp,
    to_timedelta,
    set_eng_float_format,
    options,
    Flags,
    set_option,
    NaT,
    PeriodIndex,
    Categorical,
    Interval,
    UInt8Dtype,
    UInt16Dtype,
    UInt32Dtype,
    UInt64Dtype,
    SparseDtype,
    Int8Dtype,
    Int16Dtype,
    Int32Dtype,
    Int64Dtype,
    Float32Dtype,
    Float64Dtype,
    StringDtype,
    BooleanDtype,
    CategoricalDtype,
    DatetimeTZDtype,
    IntervalDtype,
    PeriodDtype,
    RangeIndex,
    Int64Index,
    UInt64Index,
    Float64Index,
    TimedeltaIndex,
    IntervalIndex,
    IndexSlice,
    Grouper,
    array,
    Period,
    show_versions,
    DateOffset,
    timedelta_range,
    infer_freq,
    interval_range,
    ExcelWriter,
    datetime,
    NamedAgg,
    NA,
)
import os
import multiprocessing

from modin.config import Engine, Parameter

# Set this so that Pandas doesn't try to multithread by itself
os.environ["OMP_NUM_THREADS"] = "1"

_is_first_update = {}
dask_client = None
_NOINIT_ENGINES = {
    "Python",
}  # engines that don't require initialization, useful for unit tests


def _update_engine(publisher: Parameter):
    global dask_client
    from modin.config import Backend, CpuCount

    if publisher.get() == "Ray":
        from modin.engines.ray.utils import initialize_ray

        # With OmniSci backend there is only a single worker per node
        # and we allow it to work on all cores.
        if Backend.get() == "Omnisci":
            CpuCount.put(1)
            os.environ["OMP_NUM_THREADS"] = str(multiprocessing.cpu_count())
        if _is_first_update.get("Ray", True):
            initialize_ray()
    elif publisher.get() == "Dask":
        if _is_first_update.get("Dask", True):
            from modin.engines.dask.utils import initialize_dask

            initialize_dask()
    elif publisher.get() == "Cloudray":
        from modin.experimental.cloud import get_connection

        conn = get_connection()
        if _is_first_update.get("Cloudray", True):

            @conn.teleport
            def init_remote_ray(partition):
                from ray import ray_constants
                import modin
                from modin.engines.ray.utils import initialize_ray

                modin.set_backends("Ray", partition)
                initialize_ray(
                    override_is_cluster=True,
                    override_redis_address=f"localhost:{ray_constants.DEFAULT_PORT}",
                    override_redis_password=ray_constants.REDIS_DEFAULT_PASSWORD,
                )

            init_remote_ray(Backend.get())
            # import EngineDispatcher here to initialize IO class
            # so it doesn't skew read_csv() timings later on
            import modin.data_management.factories.dispatcher  # noqa: F401
        else:
            get_connection().modules["modin"].set_backends("Ray", Backend.get())
    elif publisher.get() == "Cloudpython":
        from modin.experimental.cloud import get_connection

        get_connection().modules["modin"].set_backends("Python")

    elif publisher.get() not in _NOINIT_ENGINES:
        raise ImportError("Unrecognized execution engine: {}.".format(publisher.get()))

    _is_first_update[publisher.get()] = False


from .. import __version__
from .dataframe import DataFrame
from .io import (
    read_csv,
    read_parquet,
    read_json,
    read_html,
    read_clipboard,
    read_excel,
    read_hdf,
    read_feather,
    read_stata,
    read_sas,
    read_pickle,
    read_sql,
    read_gbq,
    read_table,
    read_fwf,
    read_sql_table,
    read_sql_query,
    read_spss,
    ExcelFile,
    to_pickle,
    HDFStore,
    json_normalize,
    read_orc,
)
from .series import Series
from .general import (
    concat,
    isna,
    isnull,
    merge,
    merge_asof,
    merge_ordered,
    pivot_table,
    notnull,
    notna,
    pivot,
    to_numeric,
    to_datetime,
    unique,
    value_counts,
    get_dummies,
    melt,
    crosstab,
    lreshape,
    wide_to_long,
)
from .plotting import Plotting as plotting

__all__ = [
    "DataFrame",
    "Series",
    "read_csv",
    "read_parquet",
    "read_json",
    "read_html",
    "read_clipboard",
    "read_excel",
    "read_hdf",
    "read_feather",
    "read_stata",
    "read_sas",
    "read_pickle",
    "read_sql",
    "read_gbq",
    "read_table",
    "read_spss",
    "read_orc",
    "json_normalize",
    "concat",
    "eval",
    "cut",
    "factorize",
    "test",
    "qcut",
    "to_datetime",
    "get_dummies",
    "isna",
    "isnull",
    "merge",
    "pivot_table",
    "date_range",
    "Index",
    "MultiIndex",
    "Series",
    "bdate_range",
    "period_range",
    "DatetimeIndex",
    "to_timedelta",
    "set_eng_float_format",
    "options",
    "set_option",
    "CategoricalIndex",
    "Timedelta",
    "Timestamp",
    "NaT",
    "PeriodIndex",
    "Categorical",
    "__version__",
    "melt",
    "crosstab",
    "plotting",
    "Interval",
    "UInt8Dtype",
    "UInt16Dtype",
    "UInt32Dtype",
    "UInt64Dtype",
    "SparseDtype",
    "Int8Dtype",
    "Int16Dtype",
    "Int32Dtype",
    "Int64Dtype",
    "CategoricalDtype",
    "DatetimeTZDtype",
    "IntervalDtype",
    "PeriodDtype",
    "BooleanDtype",
    "StringDtype",
    "NA",
    "RangeIndex",
    "Int64Index",
    "UInt64Index",
    "Float64Index",
    "TimedeltaIndex",
    "IntervalIndex",
    "IndexSlice",
    "Grouper",
    "array",
    "Period",
    "show_versions",
    "DateOffset",
    "timedelta_range",
    "infer_freq",
    "interval_range",
    "ExcelWriter",
    "read_fwf",
    "read_sql_table",
    "read_sql_query",
    "ExcelFile",
    "to_pickle",
    "HDFStore",
    "lreshape",
    "wide_to_long",
    "merge_asof",
    "merge_ordered",
    "notnull",
    "notna",
    "pivot",
    "to_numeric",
    "unique",
    "value_counts",
    "datetime",
    "NamedAgg",
]

del pandas, Engine, Parameter
