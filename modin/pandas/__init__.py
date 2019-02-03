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


def initialize_ray():
    """Initializes ray based on environment variables and internal defaults."""
    if threading.current_thread().name == "MainThread":
        plasma_directory = None
        object_store_memory = None
        if "MODIN_MEMORY" in os.environ:
            object_store_memory = os.environ["MODIN_MEMORY"]
        if (
            "MODIN_OUT_OF_CORE" in os.environ
            and os.environ["MODIN_OUT_OF_CORE"].title() == "True"
        ):
            from tempfile import gettempdir

            plasma_directory = gettempdir()
            # We may have already set the memory from the environment variable, we don't
            # want to overwrite that value if we have.
            if object_store_memory is None:
                try:
                    from psutil import virtual_memory
                except ImportError:
                    raise ImportError(
                        "To use Modin out of core, please install modin[out_of_core]: "
                        '`pip install "modin[out_of_core]"`'
                    )
                # Round down to the nearest Gigabyte.
                mem_bytes = virtual_memory().total // 10 ** 9 * 10 ** 9
                # Default to 8x memory for out of core
                object_store_memory = 8 * mem_bytes
        elif "MODIN_MEMORY" in os.environ:
            object_store_memory = os.environ["MODIN_MEMORY"]
        # In case anything failed above, we can still improve the memory for Modin.
        if object_store_memory is None:
            # Round down to the nearest Gigabyte.
            object_store_memory = int(
                0.6 * ray.utils.get_system_memory() // 10 ** 9 * 10 ** 9
            )
            # If the memory pool is smaller than 2GB, just use the default in ray.
            if object_store_memory == 0:
                object_store_memory = None
        ray.init(
            redirect_output=True,
            include_webui=False,
            redirect_worker_output=True,
            ignore_reinit_error=True,
            plasma_directory=plasma_directory,
            object_store_memory=object_store_memory,
        )
        # Register custom serializer for method objects to avoid warning message.
        # We serialize `MethodType` objects when we use AxisPartition operations.
        ray.register_custom_serializer(types.MethodType, use_pickle=True)


if execution_engine == "Ray":
    initialize_ray()
    num_cpus = ray.global_state.cluster_resources()["CPU"]
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
