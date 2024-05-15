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

import warnings

import pandas
from packaging import version

__pandas_version__ = "2.2"

if (
    version.parse(pandas.__version__).release[:2]
    != version.parse(__pandas_version__).release[:2]
):
    warnings.warn(
        f"The pandas version installed ({pandas.__version__}) does not match the supported pandas version in"
        + f" Modin ({__pandas_version__}.X). This may cause undesired side effects!"
    )

# The extensions assigned to this module
_PD_EXTENSIONS_ = {}

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
        testing,
    )

import os

from modin.config import Parameter

_is_first_update = {}


def _update_engine(publisher: Parameter):
    from modin.config import (
        CpuCount,
        Engine,
        IsExperimental,
        StorageFormat,
        ValueSource,
    )

    # Set this so that Pandas doesn't try to multithread by itself
    os.environ["OMP_NUM_THREADS"] = "1"

    if publisher.get() == "Ray":
        if _is_first_update.get("Ray", True):
            from modin.core.execution.ray.common import initialize_ray

            initialize_ray()
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


from modin.pandas import arrays, errors
from modin.utils import show_versions

from .. import __version__
from .dataframe import DataFrame
from .general import (
    concat,
    crosstab,
    cut,
    get_dummies,
    isna,
    isnull,
    lreshape,
    melt,
    merge,
    merge_asof,
    merge_ordered,
    notna,
    notnull,
    pivot,
    pivot_table,
    qcut,
    to_datetime,
    to_numeric,
    to_timedelta,
    unique,
    value_counts,
    wide_to_long,
)
from .io import (
    ExcelFile,
    HDFStore,
    json_normalize,
    read_clipboard,
    read_csv,
    read_excel,
    read_feather,
    read_fwf,
    read_gbq,
    read_hdf,
    read_html,
    read_json,
    read_orc,
    read_parquet,
    read_pickle,
    read_sas,
    read_spss,
    read_sql,
    read_sql_query,
    read_sql_table,
    read_stata,
    read_table,
    read_xml,
    to_pickle,
)
from .plotting import Plotting as plotting
from .series import Series


def __getattr__(name: str):
    """
    Overrides getattr on the module to enable extensions.

    Parameters
    ----------
    name : str
        The name of the attribute being retrieved.

    Returns
    -------
    Attribute
        Returns the extension attribute, if it exists, otherwise returns the attribute
        imported in this file.
    """
    try:
        return _PD_EXTENSIONS_.get(name, globals()[name])
    except KeyError:
        raise AttributeError(f"module 'modin.pandas' has no attribute '{name}'")


__all__ = [  # noqa: F405
    "_PD_EXTENSIONS_",
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
    "errors",
]

del pandas, Parameter
