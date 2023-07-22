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
from packaging import version

__pandas_version__ = "2.0"

if (
    version.parse(pandas.__version__).release[:2]
    != version.parse(__pandas_version__).release[:2]
):
    warnings.warn(
        f"The pandas version installed ({pandas.__version__}) does not match the supported pandas version in"
        + f" Modin ({__pandas_version__}.X). This may cause undesired side effects!"
    )

# to not pollute namespace
del version

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from pandas import (
        eval,
        factorize,
        test,
        date_range,
        period_range,
        Index,
        MultiIndex,
        CategoricalIndex,
        bdate_range,
        DatetimeIndex,
        Timedelta,
        Timestamp,
        set_eng_float_format,
        options,
        describe_option,
        set_option,
        get_option,
        reset_option,
        option_context,
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
        NamedAgg,
        NA,
        api,
        ArrowDtype,
        Flags,
        Float32Dtype,
        Float64Dtype,
        from_dummies,
    )
import os

from modin.config import Parameter

_is_first_update = {}


def _update_engine(publisher: Parameter):
    from modin.config import Engine, StorageFormat, CpuCount
    from modin.config.envvars import IsExperimental
    from modin.config.pubsub import ValueSource

    # Set this so that Pandas doesn't try to multithread by itself
    os.environ["OMP_NUM_THREADS"] = "1"

    sfmt = StorageFormat.get()

    if sfmt == "Hdk":
        is_hdk = True
    elif sfmt == "Omnisci":
        is_hdk = True
        StorageFormat.put("Hdk")
        warnings.warn(
            "The OmniSci storage format has been deprecated. Please use "
            + '`StorageFormat.put("hdk")` or `MODIN_STORAGE_FORMAT="hdk"` instead.'
        )
    else:
        is_hdk = False

    if is_hdk and publisher.get_value_source() == ValueSource.DEFAULT:
        publisher.put("Native")
        IsExperimental.put(True)
    if (
        publisher.get() == "Native"
        and StorageFormat.get_value_source() == ValueSource.DEFAULT
    ):
        is_hdk = True
        StorageFormat.put("Hdk")
        IsExperimental.put(True)

    if publisher.get() == "Ray":
        if _is_first_update.get("Ray", True):
            from modin.core.execution.ray.common import initialize_ray

            initialize_ray()
    elif publisher.get() == "Native":
        # With HDK storage format there is only a single worker per node
        # and we allow it to work on all cores.
        if is_hdk:
            os.environ["OMP_NUM_THREADS"] = str(CpuCount.get())
        else:
            raise ValueError(
                f"Storage format should be 'Hdk' with 'Native' engine, but provided {sfmt}."
            )
    elif publisher.get() == "Dask":
        if _is_first_update.get("Dask", True):
            from modin.core.execution.dask.common import initialize_dask

            initialize_dask()
    elif publisher.get() == "Unidist":
        if _is_first_update.get("Unidist", True):
            from modin.core.execution.unidist.common import initialize_unidist

            initialize_unidist()
    elif publisher.get() not in Engine.NOINIT_ENGINES:
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
    notnull,
    notna,
    pivot,
    to_numeric,
    qcut,
    to_datetime,
    unique,
    value_counts,
    get_dummies,
    melt,
    crosstab,
    lreshape,
    wide_to_long,
    to_timedelta,
    pivot_table,
    cut,
)

from .plotting import Plotting as plotting
from modin.utils import show_versions

__all__ = [  # noqa: F405
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
    "describe_option",
    "set_option",
    "get_option",
    "reset_option",
    "option_context",
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
    "NamedAgg",
    "api",
    "read_xml",
    "ArrowDtype",
    "Flags",
    "Float32Dtype",
    "Float64Dtype",
    "from_dummies",
]

del pandas, Parameter
