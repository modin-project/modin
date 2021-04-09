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
This module contains the functionality that is used when benchmarking modin commits.
In the case of using utilities from the main modin code, there is a chance that when
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
    from modin.config import TestDatasetSize, AsvImplementation, Engine

    ASV_USE_IMPL = AsvImplementation.get()
    ASV_DATASET_SIZE = TestDatasetSize.get() or "Small"
    ASV_USE_ENGINE = Engine.get()
except ImportError:
    # The same benchmarking code can be run for different versions of Modin, so in
    # case of an error importing important variables, we'll just use predefined values
    ASV_USE_IMPL = os.environ.get("MODIN_ASV_USE_IMPL", "modin")
    ASV_DATASET_SIZE = os.environ.get("MODIN_TEST_DATASET_SIZE", "Small")
    ASV_USE_ENGINE = os.environ.get("MODIN_ENGINE", "Ray")

ASV_USE_IMPL = ASV_USE_IMPL.lower()
ASV_DATASET_SIZE = ASV_DATASET_SIZE.lower()
ASV_USE_ENGINE = ASV_USE_ENGINE.lower()

assert ASV_USE_IMPL in ("modin", "pandas")
assert ASV_DATASET_SIZE in ("big", "small")
assert ASV_USE_ENGINE in ("ray", "dask", "python")

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
    groupby_ngroups: str or int
        number of groups that will be used in `groupby` operation
    shape: tuple

    Return
    ------
    int
    """
    if ASV_DATASET_SIZE == "big":
        if groupby_ngroups == "huge_amount_groups":
            return min(shape[0] // 2, 5000)
        return groupby_ngroups
    else:
        return groupby_ngroups


class weakdict(dict):
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
    nrows: int
        number of rows
    ncols: int
        number of columns
    rand_low: int
        low bound for random generator
    rand_high:int
        high bound for random generator

    Return
    ------
    dict
        ncols number of keys, storing numpy.array of nrows length
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
    nrows: int
        number of rows
    ncols: int
        number of columns
    rand_low: int
        low bound for random generator
    rand_high:int
        high bound for random generator

    Return
    ------
    dict
        ncols+1 number of keys, storing numpy.array of nrows length
        one column with string values
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
    data["gb_col"] = [
        "str_{}".format(random_state.randint(rand_low, rand_high)) for i in range(nrows)
    ]
    data_cache[cache_key] = weakdict(data)
    return data


def gen_data(
    data_type: str, nrows: int, ncols: int, rand_low: int, rand_high: int
) -> dict:
    """
    Generate data with caching.

    The generated data are saved in the dictionary and on a subsequent call,
    if the keys match, saved data will be returned. Therefore, we need
    to carefully monitor the changing of saved data and make its copy if needed.

    Parameters
    ----------
    data_type: str
        type of data generation
    nrows: int
        number of rows
    ncols: int
        number of columns
    rand_low: int
        low bound for random generator
    rand_high:int
        high bound for random generator

    Return
    ------
    dict
        ncols+1 number of keys, storing numpy.array of nrows length
        one column with string values
    """
    if data_type == "int":
        return gen_int_data(nrows, ncols, rand_low, rand_high)
    elif data_type == "str_int":
        return gen_str_int_data(nrows, ncols, rand_low, rand_high)
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
    impl: str
        implementation used to create the dataframe;
        supported implemetations: {"modin", "pandas"}
    data_type: str
        type of data generation;
        supported types: {"int", "str_int"}
    nrows: int
        number of rows
    ncols: int
        number of columns
    rand_low: int
        low bound for random generator
    rand_high: int
        high bound for random generator
    groupby_ncols: int or None
        number of columns for which `groupby` will be called in the future;
        to get more stable performance results, we need to have the same number of values
        in each group every benchmarking time
    count_groups: int or None
        count of groups in groupby columns

    Return
    ------
    modin.DataFrame or pandas.DataFrame [and list of groupby columns names if
        columns for groupby were generated]
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

    Return
    ------
    str
    """
    return str(uuid.uuid4())


def random_columns(df_columns: list, columns_number: int) -> list:
    """
    Pick sublist of random columns from a given sequence.

    Parameters
    ----------
    df_columns: list
        columns to choose from
    columns_number: int
        how many columns to pick

    Return
    ------
    list
    """
    return list(random_state.choice(df_columns, size=columns_number))


def random_booleans(number: int) -> list:
    """
    Create random list of booleans with `number` elements.

    Parameters
    ----------
    number: int
        count of booleans in result list

    Return
    ------
    list
    """
    return list(random_state.choice([True, False], size=number))


def execute(df: Union[pd.DataFrame, pandas.DataFrame]):
    """
    Make sure the calculations are finished.

    Parameters
    ----------
    df: modin.DataFrame of pandas.Datarame
    """
    if ASV_USE_IMPL == "modin":
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
    shape: tuple

    Return
    ------
    str
    """
    return "_".join([str(element) for element in shape])
