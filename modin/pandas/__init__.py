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

__pandas_version__ = "1.1.2"

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
import threading
import os
import multiprocessing

from .. import execution_engine, partition_format, Publisher

# Set this so that Pandas doesn't try to multithread by itself
os.environ["OMP_NUM_THREADS"] = "1"
DEFAULT_NPARTITIONS = 4
num_cpus = 1


_is_first_update = {}
dask_client = None
_NOINIT_ENGINES = {
    "Python",
}  # engines that don't require initialization, useful for unit tests


def _update_engine(publisher: Publisher):
    global DEFAULT_NPARTITIONS, dask_client, num_cpus, partition_format

    if publisher.get() == "Ray":
        import ray
        from modin.engines.ray.utils import initialize_ray

        # With OmniSci backend there is only a single worker per node
        # and we allow it to work on all cores.
        if partition_format.get() == "Omnisci":
            os.environ["MODIN_CPUS"] = "1"
            os.environ["OMP_NUM_THREADS"] = str(multiprocessing.cpu_count())
        if _is_first_update.get("Ray", True):
            initialize_ray()
        num_cpus = ray.cluster_resources()["CPU"]
    elif publisher.get() == "Dask":  # pragma: no cover
        from distributed.client import get_client

        if threading.current_thread().name == "MainThread" and _is_first_update.get(
            "Dask", True
        ):
            import warnings

            warnings.warn("The Dask Engine for Modin is experimental.")

            try:
                dask_client = get_client()
            except ValueError:
                from distributed import Client

                num_cpus = (
                    os.environ.get("MODIN_CPUS", None) or multiprocessing.cpu_count()
                )
                dask_client = Client(n_workers=int(num_cpus))

    elif publisher.get() == "Cloudray":
        from modin.experimental.cloud import get_connection

        conn = get_connection()
        remote_ray = conn.modules["ray"]
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

            init_remote_ray(partition_format.get())
            # import EngineDispatcher here to initialize IO class
            # so it doesn't skew read_csv() timings later on
            import modin.data_management.factories.dispatcher  # noqa: F401
        else:
            get_connection().modules["modin"].set_backends(
                "Ray", partition_format.get()
            )

        num_cpus = remote_ray.cluster_resources()["CPU"]
    elif publisher.get() == "Cloudpython":
        from modin.experimental.cloud import get_connection

        get_connection().modules["modin"].set_backends("Python")

    elif publisher.get() not in _NOINIT_ENGINES:
        raise ImportError("Unrecognized execution engine: {}.".format(publisher.get()))

    _is_first_update[publisher.get()] = False
    DEFAULT_NPARTITIONS = max(4, int(num_cpus))


execution_engine.subscribe(_update_engine)

from .. import __version__
from .concat import concat
from .dataframe import DataFrame
from .datetimes import to_datetime
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
from .reshape import get_dummies, melt, crosstab, lreshape, wide_to_long
from .series import Series
from .general import (
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
    unique,
    value_counts,
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
    "DEFAULT_NPARTITIONS",
]

del pandas
