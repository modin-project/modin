from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# TODO: In the future `set_option` or similar needs to run on every node
# in order to keep all pandas instances across nodes consistent
import pandas

__pandas_version__ = "0.24.2"

if pandas.__version__ != __pandas_version__:
    raise ImportError(
        "The pandas version installed does not match the required pandas "
        "version in Modin. Please install pandas {} to use "
        "Modin.".format(__pandas_version__)
    )

from pandas import (
    eval,
    unique,
    value_counts,
    cut,
    to_numeric,
    factorize,
    test,
    qcut,
    Panel,
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
    TimeGrouper,
    Grouper,
    array,
    Period,
    show_versions,
    DateOffset,
    timedelta_range,
    infer_freq,
    interval_range,
    ExcelWriter,
    SparseArray,
    SparseSeries,
    SparseDataFrame,
    datetime,
)
import threading
import os
import ray
import types

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
    read_msgpack,
    read_stata,
    read_sas,
    read_pickle,
    read_sql,
    read_gbq,
    read_table,
    read_fwf,
    read_sql_table,
    read_sql_query,
    ExcelFile,
    to_pickle,
    HDFStore,
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
)
from .plotting import Plotting as plotting
from .. import __execution_engine__ as execution_engine

# Set this so that Pandas doesn't try to multithread by itself
os.environ["OMP_NUM_THREADS"] = "1"
num_cpus = 1


def initialize_ray():
    """Initializes ray based on environment variables and internal defaults."""
    if threading.current_thread().name == "MainThread":
        plasma_directory = None
        cluster = os.environ.get("MODIN_RAY_CLUSTER", None)
        redis_address = os.environ.get("MODIN_REDIS_ADDRESS", None)
        if cluster == "True" and redis_address is not None:
            # We only start ray in a cluster setting for the head node.
            ray.init(
                include_webui=False,
                ignore_reinit_error=True,
                redis_address=redis_address,
                logging_level=100,
            )
        elif cluster is None:
            object_store_memory = os.environ.get("MODIN_MEMORY", None)
            if os.environ.get("MODIN_OUT_OF_CORE", "False").title() == "True":
                from tempfile import gettempdir

                plasma_directory = gettempdir()
                # We may have already set the memory from the environment variable, we don't
                # want to overwrite that value if we have.
                if object_store_memory is None:
                    # Round down to the nearest Gigabyte.
                    mem_bytes = ray.utils.get_system_memory() // 10 ** 9 * 10 ** 9
                    # Default to 8x memory for out of core
                    object_store_memory = 8 * mem_bytes
            # In case anything failed above, we can still improve the memory for Modin.
            if object_store_memory is None:
                # Round down to the nearest Gigabyte.
                object_store_memory = int(
                    0.6 * ray.utils.get_system_memory() // 10 ** 9 * 10 ** 9
                )
                # If the memory pool is smaller than 2GB, just use the default in ray.
                if object_store_memory == 0:
                    object_store_memory = None
            else:
                object_store_memory = int(object_store_memory)
            ray.init(
                include_webui=False,
                ignore_reinit_error=True,
                plasma_directory=plasma_directory,
                object_store_memory=object_store_memory,
                redis_address=redis_address,
                logging_level=100,
            )
        # Register custom serializer for method objects to avoid warning message.
        # We serialize `MethodType` objects when we use AxisPartition operations.
        ray.register_custom_serializer(types.MethodType, use_pickle=True)


if execution_engine == "Ray":
    initialize_ray()
    num_cpus = ray.cluster_resources()["CPU"]
elif execution_engine == "Dask":  # pragma: no cover
    from distributed.client import _get_global_client

    if threading.current_thread().name == "MainThread":
        # initialize the dask client
        client = _get_global_client()
        if client is None:
            from distributed import Client

            client = Client()
        num_cpus = sum(client.ncores().values())
elif execution_engine != "Python":
    raise ImportError("Unrecognized execution engine: {}.".format(execution_engine))

DEFAULT_NPARTITIONS = max(4, int(num_cpus))

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
    "read_msgpack",
    "read_stata",
    "read_sas",
    "read_pickle",
    "read_sql",
    "read_gbq",
    "read_table",
    "concat",
    "eval",
    "unique",
    "value_counts",
    "cut",
    "to_numeric",
    "factorize",
    "test",
    "qcut",
    "to_datetime",
    "get_dummies",
    "isna",
    "isnull",
    "merge",
    "pivot_table",
    "Panel",
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
    "RangeIndex",
    "Int64Index",
    "UInt64Index",
    "Float64Index",
    "TimedeltaIndex",
    "IntervalIndex",
    "IndexSlice",
    "TimeGrouper",
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
    "SparseArray",
    "SparseSeries",
    "SparseDataFrame",
    "datetime",
]

del pandas
