from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# TODO: In the future `set_option` or similar needs to run on every node
# in order to keep all pandas instances across nodes consistent
import pandas
from pandas import (
    eval,
    unique,
    value_counts,
    cut,
    to_numeric,
    factorize,
    test,
    qcut,
    match,
    Panel,
    date_range,
    period_range,
    Index,
    MultiIndex,
    CategoricalIndex,
    Series,
    bdate_range,
    DatetimeIndex,
    Timedelta,
    Timestamp,
    to_timedelta,
    set_eng_float_format,
    set_option,
    NaT,
    PeriodIndex,
    Categorical,
)
import threading
import os
import ray

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
)
from .reshape import get_dummies, melt, crosstab
from .general import isna, isnull, merge, pivot_table
from .plotting import Plotting as plotting
from .. import __execution_engine__ as execution_engine

__pandas_version__ = "0.23.4"

if pandas.__version__ != __pandas_version__:
    raise ImportError(
        "The pandas version installed does not match the required pandas "
        "version in Modin. Please install pandas {} to use "
        "Modin.".format(__pandas_version__)
    )

# Set this so that Pandas doesn't try to multithread by itself
os.environ["OMP_NUM_THREADS"] = "1"
num_cpus = 1

if execution_engine == "Ray":
    try:
        if threading.current_thread().name == "MainThread":
            ray.init(
                redirect_output=True,
                include_webui=False,
                redirect_worker_output=True,
                ignore_reinit_error=True,
            )
            num_cpus = ray.global_state.cluster_resources()["CPU"]
    except AssertionError:
        pass
elif execution_engine == "Dask":
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
    "match",
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
]

del pandas
