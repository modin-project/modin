from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# TODO: In the future `set_option` or similar needs to run on every node
# in order to keep all pandas instances across nodes consistent
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

from .. import __git_revision__, __version__
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
)
from .reshape import get_dummies

# Set this so that Pandas doesn't try to multithread by itself
os.environ["OMP_NUM_THREADS"] = "1"

try:
    if threading.current_thread().name == "MainThread":
        ray.init(
            redirect_output=True,
            include_webui=False,
            redirect_worker_output=True,
            use_raylet=True,
        )
except AssertionError:
    pass

num_cpus = ray.global_state.cluster_resources()["CPU"]
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
    "Panel",
    "date_range",
    "Index",
    "MultiIndex",
    "Series",
    "bdate_range",
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
    "__git_revision__",
    "__version__",
]
