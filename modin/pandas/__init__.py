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
import warnings

__pandas_version__ = "1.4.2"

if pandas.__version__ != __pandas_version__:
    warnings.warn(
        f"The pandas version installed {pandas.__version__} does not match the supported pandas version in"
        + f" Modin {__pandas_version__}. This may cause undesired side effects!"
    )

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
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
        DateOffset,
        timedelta_range,
        infer_freq,
        interval_range,
        ExcelWriter,
        datetime,
        NamedAgg,
        NA,
        api,
    )
import os
import multiprocessing

from modin.config import Engine, Parameter

# Set this so that Pandas doesn't try to multithread by itself
os.environ["OMP_NUM_THREADS"] = "1"

_is_first_update = {}
_NOINIT_ENGINES = {
    "Python",
}  # engines that don't require initialization, useful for unit tests


def _update_engine(publisher: Parameter):
    from modin.config import StorageFormat, CpuCount
    from modin.config.envvars import IsExperimental
    from modin.config.pubsub import ValueSource

    if (
        StorageFormat.get() == "Omnisci"
        and publisher.get_value_source() == ValueSource.DEFAULT
    ):
        publisher.put("Native")
        IsExperimental.put(True)
    if (
        publisher.get() == "Native"
        and StorageFormat.get_value_source() == ValueSource.DEFAULT
    ):
        StorageFormat.put("Omnisci")
        IsExperimental.put(True)

    if publisher.get() == "Ray":
        if _is_first_update.get("Ray", True):
            from modin.core.execution.ray.common import initialize_ray

            initialize_ray()
    elif publisher.get() == "Native":
        # With OmniSci storage format there is only a single worker per node
        # and we allow it to work on all cores.
        if StorageFormat.get() == "Omnisci":
            os.environ["OMP_NUM_THREADS"] = str(CpuCount.get())
        else:
            raise ValueError(
                f"Storage format should be 'Omnisci' with 'Native' engine, but provided {StorageFormat.get()}."
            )
    elif publisher.get() == "Dask":
        if _is_first_update.get("Dask", True):
            from modin.core.execution.dask.common.utils import initialize_dask

            initialize_dask()
    elif publisher.get() == "Cloudray":
        from modin.experimental.cloud import get_connection

        conn = get_connection()
        if _is_first_update.get("Cloudray", True):

            @conn.teleport
            def init_remote_ray(partition):
                from ray import ray_constants
                import modin
                from modin.core.execution.ray.common import initialize_ray

                modin.set_execution("Ray", partition)
                initialize_ray(
                    override_is_cluster=True,
                    override_redis_address=f"localhost:{ray_constants.DEFAULT_PORT}",
                    override_redis_password=ray_constants.REDIS_DEFAULT_PASSWORD,
                )

            init_remote_ray(StorageFormat.get())
            # import FactoryDispatcher here to initialize IO class
            # so it doesn't skew read_csv() timings later on
            import modin.core.execution.dispatching.factories.dispatcher  # noqa: F401
        else:
            get_connection().modules["modin"].set_execution("Ray", StorageFormat.get())
    elif publisher.get() == "Cloudpython":
        from modin.experimental.cloud import get_connection

        get_connection().modules["modin"].set_execution("Python")
    elif publisher.get() == "Cloudnative":
        from modin.experimental.cloud import get_connection

        assert (
            StorageFormat.get() == "Omnisci"
        ), f"Storage format should be 'Omnisci' with 'Cloudnative' engine, but provided {StorageFormat.get()}."
        get_connection().modules["modin"].set_execution("Native", "OmniSci")

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
    read_xml,
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
from modin.utils import show_versions

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
    "api",
]

del pandas, Engine, Parameter
