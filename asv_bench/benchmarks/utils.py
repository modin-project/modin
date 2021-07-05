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

"""
The module contains the functionality that is used when benchmarking Modin commits.

In the case of using utilities from the main Modin code, there is a chance that when
benchmarking old commits, the utilities changed, which in turn can unexpectedly affect
the performance results, hence some utility functions are duplicated here.
"""

import os
import logging
import modin.pandas as pd
import pandas
import numpy as np
import uuid
from typing import Optional, Union

RAND_LOW = 0
RAND_HIGH = 100
random_state = np.random.RandomState(seed=42)


try:
    from modin.config import NPartitions

    NPARTITIONS = NPartitions.get()
except ImportError:
    NPARTITIONS = pd.DEFAULT_NPARTITIONS

try:
    from modin.config import TestDatasetSize, AsvImplementation, Engine, Backend

    ASV_USE_IMPL = AsvImplementation.get()
    ASV_DATASET_SIZE = TestDatasetSize.get() or "Small"
    ASV_USE_ENGINE = Engine.get()
    ASV_USE_BACKEND = Backend.get()
except ImportError:
    # The same benchmarking code can be run for different versions of Modin, so in
    # case of an error importing important variables, we'll just use predefined values
    ASV_USE_IMPL = os.environ.get("MODIN_ASV_USE_IMPL", "modin")
    ASV_DATASET_SIZE = os.environ.get("MODIN_TEST_DATASET_SIZE", "Small")
    ASV_USE_ENGINE = os.environ.get("MODIN_ENGINE", "Ray")
    ASV_USE_BACKEND = os.environ.get("MODIN_BACKEND", "Pandas")

ASV_USE_IMPL = ASV_USE_IMPL.lower()
ASV_DATASET_SIZE = ASV_DATASET_SIZE.lower()
ASV_USE_ENGINE = ASV_USE_ENGINE.lower()
ASV_USE_BACKEND = ASV_USE_BACKEND.lower()

assert ASV_USE_IMPL in ("modin", "pandas")
assert ASV_DATASET_SIZE in ("big", "small")
assert ASV_USE_ENGINE in ("ray", "dask", "python")
assert ASV_USE_BACKEND in ("pandas", "omnisci", "pyarrow")

BINARY_OP_DATA_SIZE = {
    "big": [
        ((5000, 5000), (5000, 5000)),
        # the case extremely inefficient
        # ((20, 500_000), (10, 1_000_000)),
        ((500_000, 20), (1_000_000, 10)),
    ],
    "small": [
        ((250, 250), (250, 250)),
        ((20, 10_000), (10, 25_000)),
        ((10_000, 20), (25_000, 10)),
    ],
}

UNARY_OP_DATA_SIZE = {
    "big": [
        (5000, 5000),
        # the case extremely inefficient
        # (10, 1_000_000),
        (1_000_000, 10),
    ],
    "small": [
        (250, 250),
        (10, 10_000),
        (10_000, 10),
    ],
}

GROUPBY_NGROUPS = {
    "big": [100, "huge_amount_groups"],
    "small": [5],
}

IMPL = {
    "modin": pd,
    "pandas": pandas,
}


def translator_groupby_ngroups(groupby_ngroups: Union[str, int], shape: tuple) -> int:
    """
    Translate a string representation of the number of groups, into a number.

    Parameters
    ----------
    groupby_ngroups : str or int
        Number of groups that will be used in `groupby` operation.
    shape : tuple
        Same as pandas.Dataframe.shape.

    Returns
    -------
    int
    """
    if ASV_DATASET_SIZE == "big":
        if groupby_ngroups == "huge_amount_groups":
            return min(shape[0] // 2, 5000)
        return groupby_ngroups
    else:
        return groupby_ngroups


class weakdict(dict):  # noqa: GL08
    __slots__ = ("__weakref__",)


data_cache = dict()
dataframes_cache = dict()


def gen_int_data(nrows: int, ncols: int, rand_low: int, rand_high: int) -> dict:
    """
    Generate int data with caching.

    The generated data are saved in the dictionary and on a subsequent call,
    if the keys match, saved data will be returned. Therefore, we need
    to carefully monitor the changing of saved data and make its copy if needed.

    Parameters
    ----------
    nrows : int
        Number of rows.
    ncols : int
        Number of columns.
    rand_low : int
        Low bound for random generator.
    rand_high : int
        High bound for random generator.

    Returns
    -------
    dict
        Number of keys - `ncols`, each of them store np.ndarray of `nrows` length.
    """
    cache_key = ("int", nrows, ncols, rand_low, rand_high)
    if cache_key in data_cache:
        return data_cache[cache_key]

    logging.info(
        "Generating int data {} rows and {} columns [{}-{}]".format(
            nrows, ncols, rand_low, rand_high
        )
    )
    data = {
        "col{}".format(i): random_state.randint(rand_low, rand_high, size=(nrows))
        for i in range(ncols)
    }
    data_cache[cache_key] = weakdict(data)
    return data


def gen_str_int_data(nrows: int, ncols: int, rand_low: int, rand_high: int) -> dict:
    """
    Generate int data and string data with caching.

    The generated data are saved in the dictionary and on a subsequent call,
    if the keys match, saved data will be returned. Therefore, we need
    to carefully monitor the changing of saved data and make its copy if needed.

    Parameters
    ----------
    nrows : int
        Number of rows.
    ncols : int
        Number of columns.
    rand_low : int
        Low bound for random generator.
    rand_high : int
        High bound for random generator.

    Returns
    -------
    dict
        Number of keys - `ncols`, each of them store np.ndarray of `nrows` length.
        One of the columns with string values.
    """
    cache_key = ("str_int", nrows, ncols, rand_low, rand_high)
    if cache_key in data_cache:
        return data_cache[cache_key]

    logging.info(
        "Generating str_int data {} rows and {} columns [{}-{}]".format(
            nrows, ncols, rand_low, rand_high
        )
    )
    data = gen_int_data(nrows, ncols, rand_low, rand_high).copy()
    # convert values in arbitary column to string type
    key = list(data.keys())[0]
    data[key] = [f"str_{x}" for x in data[key]]
    data_cache[cache_key] = weakdict(data)
    return data


def gen_true_false_data(nrows: int, ncols: int, true_false_values: list):
    """
    Generate data with "true" and "false" values with caching.

    The generated data are saved in the dictionary and on a subsequent call,
    if the keys match, saved data will be returned. Therefore, we need
    to carefully monitor the changing of saved data and make its copy if needed.

    Parameters
    ----------
    nrows : int
        Number of rows.
    ncols : int
        Number of columns.
    true_false_values : list
        List with strings that represent "true" and "false" values.

    Returns
    -------
    dict
        Number of keys - `ncols`, each of them store np.ndarray of `nrows` length.
    """
    cache_key = ("true_false", nrows, ncols, "_".join(true_false_values))
    if cache_key in data_cache:
        return data_cache[cache_key]

    logging.info(
        "Generating true_false data {} rows and {} columns with values {}".format(
            nrows, ncols, true_false_values
        )
    )
    data = {
        "tf_col{}".format(i): random_state.choice(true_false_values, size=(nrows))
        for i in range(ncols)
    }
    data_cache[cache_key] = weakdict(data)
    return data


def gen_true_false_int_data(nrows, ncols, rand_low, rand_high):
    """
    Generate int data and string data "true" and "false" values with caching.

    The generated data are saved in the dictionary and on a subsequent call,
    if the keys match, saved data will be returned. Therefore, we need
    to carefully monitor the changing of saved data and make its copy if needed.

    Parameters
    ----------
    nrows : int
        Number of rows.
    ncols : int
        Number of columns.
    rand_low : int
        Low bound for random generator.
    rand_high : int
        High bound for random generator.

    Returns
    -------
    dict
        Number of keys - `ncols`, each of them store np.ndarray of `nrows` length.
        One half of the columns with integer values, another half - with "true" and
        "false" string values.
    """
    cache_key = ("true_false_int", nrows, ncols, rand_low, rand_high)
    if cache_key in data_cache:
        return data_cache[cache_key]

    logging.info(
        "Generating true_false_int data {} rows and {} columns [{}-{}]".format(
            nrows, ncols, rand_low, rand_high
        )
    )
    data = gen_int_data(nrows // 2, ncols // 2, rand_low, rand_high).copy()
    data_true_false = gen_true_false_data(
        nrows - nrows // 2, ncols - ncols // 2, ["Yes", "true", "No", "false"]
    ).copy()
    data.update(data_true_false)
    data_cache[cache_key] = weakdict(data)
    return data


def gen_data(
    data_type: str,
    nrows: int,
    ncols: int,
    rand_low: int,
    rand_high: int,
) -> dict:
    """
    Generate data with caching.

    The generated data are saved in the dictionary and on a subsequent call,
    if the keys match, saved data will be returned. Therefore, we need
    to carefully monitor the changing of saved data and make its copy if needed.

    Parameters
    ----------
    data_type : {"int", "str_int", "true_false_int"}
        Type of data generation.
    nrows : int
        Number of rows.
    ncols : int
        Number of columns.
    rand_low : int
        Low bound for random generator.
    rand_high : int
        High bound for random generator.

    Returns
    -------
    dict
        Number of keys - `ncols`, each of them store np.ndarray of `nrows` length.

    Notes
    -----
    Returned data type depends on the `data_type` parameter in the next way:

    - `data_type`=="int" - all columns will be contain only integer values
    - `data_type`=="str_int" some of the columns will be of string type,
    - `data_type`=="true_false_int" half of the columns will be filed with
      string values representing "true" and "false" values and another half - with
      integers.
    """
    if data_type == "int":
        return gen_int_data(nrows, ncols, rand_low, rand_high)
    elif data_type == "str_int":
        return gen_str_int_data(nrows, ncols, rand_low, rand_high)
    elif data_type == "true_false_int":
        return gen_true_false_int_data(nrows, ncols, rand_low, rand_high)
    else:
        assert False


def generate_dataframe(
    impl: str,
    data_type: str,
    nrows: int,
    ncols: int,
    rand_low: int,
    rand_high: int,
    groupby_ncols: Optional[int] = None,
    count_groups: Optional[int] = None,
) -> Union[pd.DataFrame, pandas.DataFrame]:
    """
    Generate DataFrame with caching.

    The generated dataframes are saved in the dictionary and on a subsequent call,
    if the keys match, one of the saved dataframes will be returned. Therefore, we need
    to carefully monitor that operations that change the dataframe work with its copy.

    Parameters
    ----------
    impl : str
        Implementation used to create the dataframe;
        supported implemetations: {"modin", "pandas"}.
    data_type : str
        Type of data generation;
        supported types: {"int", "str_int"}.
    nrows : int
        Number of rows.
    ncols : int
        Number of columns.
    rand_low : int
        Low bound for random generator.
    rand_high : int
        High bound for random generator.
    groupby_ncols : int, default: None
        Number of columns for which `groupby` will be called in the future;
        to get more stable performance results, we need to have the same number of values
        in each group every benchmarking time.
    count_groups : int, default: None
        Count of groups in groupby columns.

    Returns
    -------
    modin.pandas.DataFrame or pandas.DataFrame [and list]

    Notes
    -----
    The list of groupby columns names returns when groupby columns are generated.
    """
    assert not (
        (groupby_ncols is None) ^ (count_groups is None)
    ), "You must either specify both parameters 'groupby_ncols' and 'count_groups' or none of them."

    if groupby_ncols and count_groups:
        ncols -= groupby_ncols

    cache_key = (
        impl,
        data_type,
        nrows,
        ncols,
        rand_low,
        rand_high,
        groupby_ncols,
        count_groups,
    )

    if cache_key in dataframes_cache:
        return dataframes_cache[cache_key]

    logging.info(
        "Allocating {} DataFrame {}: {} rows and {} columns [{}-{}]".format(
            impl, data_type, nrows, ncols, rand_low, rand_high
        )
    )
    data = gen_data(data_type, nrows, ncols, rand_low, rand_high)

    if groupby_ncols and count_groups:
        groupby_columns = [f"groupby_col{x}" for x in range(groupby_ncols)]
        for groupby_col in groupby_columns:
            data[groupby_col] = np.tile(np.arange(count_groups), nrows // count_groups)

    if impl == "modin":
        df = pd.DataFrame(data)
    elif impl == "pandas":
        df = pandas.DataFrame(data)
    else:
        assert False

    if groupby_ncols and count_groups:
        dataframes_cache[cache_key] = df, groupby_columns
        return df, groupby_columns

    dataframes_cache[cache_key] = df
    return df


def random_string() -> str:
    """
    Create a 36-character random string.

    Returns
    -------
    str
    """
    return str(uuid.uuid4())


def random_columns(df_columns: list, columns_number: int) -> list:
    """
    Pick sublist of random columns from a given sequence.

    Parameters
    ----------
    df_columns : list
        Columns to choose from.
    columns_number : int
        How many columns to pick.

    Returns
    -------
    list
    """
    return list(random_state.choice(df_columns, size=columns_number))


def random_booleans(number: int) -> list:
    """
    Create random list of booleans with `number` elements.

    Parameters
    ----------
    number : int
        Count of booleans in result list.

    Returns
    -------
    list
    """
    return list(random_state.choice([True, False], size=number))


def execute(df: Union[pd.DataFrame, pandas.DataFrame]):
    """
    Make sure the calculations are finished.

    Parameters
    ----------
    df : modin.pandas.DataFrame or pandas.Datarame
    """
    if ASV_USE_IMPL == "modin":
        if ASV_USE_BACKEND == "omnisci":
            df._query_compiler._modin_frame._execute()
            return
        partitions = df._query_compiler._modin_frame._partitions
        all(
            map(
                lambda partition: partition.drain_call_queue() or True,
                partitions.flatten(),
            )
        )
        if ASV_USE_ENGINE == "ray":
            from ray import wait

            all(map(lambda partition: wait([partition.oid]), partitions.flatten()))
        elif ASV_USE_ENGINE == "dask":
            from dask.distributed import wait

            all(map(lambda partition: wait(partition.future), partitions.flatten()))
        elif ASV_USE_ENGINE == "python":
            pass

    elif ASV_USE_IMPL == "pandas":
        pass


def get_shape_id(shape: tuple) -> str:
    """
    Join shape numbers into a string with `_` delimiters.

    Parameters
    ----------
    shape : tuple
        Same as pandas.Dataframe.shape.

    Returns
    -------
    str
    """
    return "_".join([str(element) for element in shape])


def prepare_io_data(test_filename: str, data_type: str):
    """
    Prepare data for IO tests with caching.

    Parameters
    ----------
    test_filename : str
        Unique file identifier that is used to distinguish data
        for different tests.
    data_type : {"int", "str_int", "true_false_int"}
        Type of data generation.

    Returns
    -------
    test_filenames : dict
        Dcitionary that maps dataset shape to the file on disk.
    """
    test_filenames = {}
    for shape in UNARY_OP_DATA_SIZE[ASV_DATASET_SIZE]:
        shape_id = get_shape_id(shape)
        test_filenames[shape_id] = f"{test_filename}_{shape_id}.csv"
        df = generate_dataframe("pandas", data_type, *shape, RAND_LOW, RAND_HIGH)
        df.to_csv(test_filenames[shape_id], index=False)

    return test_filenames
