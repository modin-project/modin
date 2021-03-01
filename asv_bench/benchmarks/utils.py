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

import os
import logging
import modin.pandas as pd
import pandas
import numpy as np
import uuid

RAND_LOW = 0
RAND_HIGH = 100
random_state = np.random.RandomState(seed=42)


try:
    from modin.config import NPartitions

    NPARTITIONS = NPartitions.get()
except ImportError:
    NPARTITIONS = pd.DEFAULT_NPARTITIONS

try:
    from modin.config import TestDatasetSize, AsvImplementation

    ASV_USE_IMPL = AsvImplementation.get()
    ASV_DATASET_SIZE = TestDatasetSize.get() or "Small"
except ImportError:
    # The same benchmarking code can be run for different versions of Modin, so in
    # case of an error importing important variables, we'll just use predefined values
    ASV_USE_IMPL = os.environ.get("MODIN_ASV_USE_IMPL", "modin")
    ASV_DATASET_SIZE = os.environ.get("MODIN_TEST_DATASET_SIZE", "Small")

assert ASV_USE_IMPL in ("modin", "pandas")

BINARY_OP_DATA_SIZE = {
    "Big": [
        ((5000, 5000), (5000, 5000)),
        # the case extremely inefficient
        # ((20, 500_000), (10, 1_000_000)),
        ((500_000, 20), (1_000_000, 10)),
    ],
    "Small": [
        ((250, 250), (250, 250)),
        ((20, 10_000), (10, 25_000)),
        ((10_000, 20), (25_000, 10)),
    ],
}

UNARY_OP_DATA_SIZE = {
    "Big": [
        (5000, 5000),
        # the case extremely inefficient
        # (10, 1_000_000),
        (1_000_000, 10),
    ],
    "Small": [
        (250, 250),
        (10, 10_000),
        (10_000, 10),
    ],
}

GROUPBY_NGROUPS = {
    "Big": [100, "huge_amount_groups"],
    "Small": [5],
}

IMPL = {
    "modin": pd,
    "pandas": pandas,
}


def translator_groupby_ngroups(groupby_ngroups, shape):
    if ASV_DATASET_SIZE == "Big":
        if groupby_ngroups == "huge_amount_groups":
            return min(shape[0] // 2, 5000)
        return groupby_ngroups
    else:
        return groupby_ngroups


class weakdict(dict):
    __slots__ = ("__weakref__",)


data_cache = dict()
dataframes_cache = dict()


def gen_int_data(nrows, ncols, rand_low, rand_high):
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


def gen_str_int_data(nrows, ncols, rand_low, rand_high):
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


def gen_data(data_type, nrows, ncols, rand_low, rand_high):
    if data_type == "int":
        return gen_int_data(nrows, ncols, rand_low, rand_high)
    elif data_type == "str_int":
        return gen_str_int_data(nrows, ncols, rand_low, rand_high)
    else:
        assert False


def generate_dataframe(
    impl,
    data_type,
    nrows,
    ncols,
    rand_low,
    rand_high,
    groupby_ncols=None,
    count_groups=None,
):
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
    else:
        cache_key = (impl, data_type, nrows, ncols, rand_low, rand_high)

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


def random_string():
    return str(uuid.uuid1())


def random_columns(df_columns, columns_number):
    return list(random_state.choice(df_columns, size=columns_number))


def random_booleans(number):
    return list(random_state.choice([True, False], size=number))


def execute(df):
    "Make sure the calculations are done."
    return df.shape, df.dtypes


def get_shape_id(array):
    return "_".join([str(element) for element in array])
