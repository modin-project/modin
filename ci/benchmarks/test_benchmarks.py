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

import logging
import pytest
import pandas
import modin.pandas as pd
from modin.pandas.test.utils import random_state

RAND_LOW = 0
RAND_HIGH = 100


class weakdict(dict):
    __slots__ = ("__weakref__",)


# TODO: Using weak reference cache results in data object being always
# collected which defeats the purpose of a cache. Maybe there is some
# workaround for this which would allow objects to remain in
# memory. Currently usage of weak reference dictionaries is turned off
# because it slows benchmark down since objects have to be always
# generated again.
#
# import weakref
# data_cache = weakref.WeakValueDictionary()
# dataframes_cache = weakref.WeakValueDictionary()

data_cache = dict()
dataframes_cache = dict()


def gen_int_data(ncols, nrows, rand_low, rand_high):
    cache_key = ("int", ncols, nrows, rand_low, rand_high)
    if cache_key in data_cache:
        return data_cache[cache_key]

    logging.info(
        "Generating int data {} rows and {} columns [{}-{}]".format(nrows, ncols, rand_low, rand_high)
    )
    data = {
        "col{}".format(i): random_state.randint(rand_low, rand_high, size=(nrows))
        for i in range(ncols)
    }
    data_cache[cache_key] = weakdict(data)
    return data


def gen_str_int_data(ncols, nrows, rand_low, rand_high):
    cache_key = ("str_int", ncols, nrows, rand_low, rand_high)
    if cache_key in data_cache:
        return data_cache[cache_key]

    logging.info(
        "Generating str_int data {} rows and {} columns [{}-{}]".format(
            nrows, ncols, rand_low, rand_high
        )
    )
    data = gen_int_data(ncols, nrows, rand_low, rand_high).copy()
    data["gb_col"] = [
        "str_{}".format(random_state.randint(rand_low, rand_high)) for i in range(nrows)
    ]
    data_cache[cache_key] = weakdict(data)
    return data


def gen_data(data_type, ncols, nrows, rand_low, rand_high):
    if data_type == "int":
        return gen_int_data(ncols, nrows, rand_low, rand_high)
    elif data_type == "str_int":
        return gen_str_int_data(ncols, nrows, rand_low, rand_high)
    else:
        assert False


def generate_dataframe(impl, data_type, ncols, nrows, rand_low, rand_high):
    cache_key = (impl, data_type, ncols, nrows, rand_low, rand_high)
    if cache_key in dataframes_cache:
        return dataframes_cache[cache_key]

    logging.info(
        "Allocating {} DataFrame {}: {} rows and {} columns [{}-{}]".format(
            impl, data_type, nrows, ncols, rand_low, rand_high
        )
    )
    data = gen_data(data_type, ncols, nrows, rand_low, rand_high)
    if impl == "modin":
        df = pd.DataFrame(data)
    elif impl == "pandas":
        df = pandas.DataFrame(data)
    else:
        assert False
    dataframes_cache[cache_key] = df
    return df


################ DataFrame.sum and DataFrame.agg(sum) ################


def benchmark_sum_function(df):
    s = df.sum()
    string = "%s" % repr(s)
    return string


def benchmark_agg_sum_function(df):
    s = df.agg(sum)
    string = "%s" % repr(s)
    return string


def benchmark_agg_sum_string_function(df):
    s = df.agg(sum)
    string = "%s" % repr(s)
    return string


@pytest.mark.parametrize("impl", ["modin", "pandas"])
@pytest.mark.parametrize("data_type", ["int"])
@pytest.mark.parametrize(
    "data_size", [(10_000, 10_000)], ids=lambda t: "{}x{}".format(t[0], t[1])
)
@pytest.mark.parametrize(
    "func",
    [
        benchmark_sum_function,
        benchmark_agg_sum_function,
        benchmark_agg_sum_string_function,
    ],
    ids=["sum", "agg(sum)", 'agg("sum")'],
)
def test_sum(benchmark, impl, data_type, data_size, func):
    result = benchmark.pedantic(
        func,
        kwargs={
            "df": generate_dataframe(
                impl, data_type, data_size[0], data_size[1], RAND_LOW, RAND_HIGH
            )
        },
        iterations=5,
        rounds=10,
        warmup_rounds=2,
    )
    return result


################ GroupBy.sum and GroupBy.agg(sum) ################

RAND_LOW = 0
RAND_HIGH = 100000


def benchmark_groupby_sum_function(gb):
    result = gb.sum()
    string = "%s" % repr(result)
    return string


def benchmark_groupby_agg_sum_function(gb):
    result = gb.agg(sum)
    string = "%s" % repr(result)
    return string


@pytest.mark.parametrize("impl", ["modin", "pandas"])
@pytest.mark.parametrize("data_type", ["int", "str_int"])
@pytest.mark.parametrize(
    "data_size",
    [(10_000, 10_000), (10, 10_000_000)],
    ids=lambda t: "{}x{}".format(t[0], t[1]),
)
@pytest.mark.parametrize(
    "index_col", [None, "col1"], ids=["no_indexing", "with_indexing"]
)
@pytest.mark.parametrize(
    "func",
    [
        benchmark_groupby_sum_function,
        benchmark_groupby_agg_sum_function,
    ],
    ids=["sum", "agg(sum)"],
)
def test_groupby_sum(benchmark, impl, data_type, data_size, index_col, func):
    df = generate_dataframe(
        impl, data_type, data_size[0], data_size[1], RAND_LOW, RAND_HIGH
    )
    if data_type == "int":
        gb1 = df.groupby("col0")
    else:
        gb1 = df.groupby("gb_col")
    if index_col is not None:
        gb2 = gb1[index_col]
    else:
        gb2 = gb1

    result = benchmark.pedantic(
        func,
        kwargs={"gb": gb2},
        iterations=5,
        rounds=10,
        warmup_rounds=2,
    )
    return result

################ DataFrame == 0 ################

def benchmark_generate_mask(df):
    s = df == 0
    return s


@pytest.mark.parametrize("impl", ["modin", "pandas"])
@pytest.mark.parametrize("data_type", ["int"])
@pytest.mark.parametrize(
    "data_size",
    [(10_000, 10_000), (10, 10_000_000), (10_000_000, 10)],
    ids=lambda t: "{}x{}".format(t[0], t[1]),
)
@pytest.mark.parametrize(
    "index_col", [None, "col1"], ids=["no_indexing", "with_indexing"]
)
def test_mask(benchmark, impl, data_type, data_size, index_col):
    df1 = generate_dataframe(
        impl, data_type, data_size[0], data_size[1], RAND_LOW, RAND_HIGH
    )
    if index_col is not None:
        df2 = df1[index_col]
    else:
        df2 = df1

    result = benchmark.pedantic(
        benchmark_generate_mask,
        kwargs={"df": df2}
    )
    return result

################ DataFrame[mask] ################

def benchmark_mask_index(df, mask):
    s = df[mask]
    return s


@pytest.mark.parametrize("impl", ["modin", "pandas"])
@pytest.mark.parametrize("data_type", ["int"])
@pytest.mark.parametrize(
    "data_size",
    [(10_000, 10_000), (10, 10_000_000), (1_000_000, 10)],
    ids=lambda t: "{}x{}".format(t[0], t[1]),
)
def test_mask_index(benchmark, impl, data_type, data_size):
    df = generate_dataframe(
        impl, data_type, data_size[0], data_size[1], RAND_LOW, RAND_HIGH
    )
    mask = df[df.columns[0]] == 0

    result = benchmark.pedantic(
        benchmark_mask_index,
        kwargs={"df": df, "mask": mask}
    )
    return result

################ DataFrame.merge ################

def benchmark_merge(df1, df2, how, sort):
    s = df1.merge(df2, on=df1.columns[0], how=how, sort=sort)
    return s


@pytest.mark.parametrize("impl", ["modin", "pandas"])
@pytest.mark.parametrize("data_type", ["int"])
@pytest.mark.parametrize(
    "data_size",
    [(5000, 5000, 5000, 5000), (10, 1_000_000, 10, 1_000_000), (1_000_000, 10, 1_000_000, 10)],
    ids=lambda t: "{}x{} and {}x{}".format(t[0], t[1], t[2], t[3]),
)
@pytest.mark.parametrize("how", ["left", "right", "outer", "inner"])
@pytest.mark.parametrize("sort", [False, True])
def test_merge(benchmark, impl, data_type, data_size, how, sort):
    df1 = generate_dataframe(
        impl, data_type, data_size[0], data_size[1], RAND_LOW, RAND_HIGH
    )
    df2 = generate_dataframe(
        impl, data_type, data_size[2], data_size[3], RAND_LOW, RAND_HIGH
    )

    result = benchmark(
        benchmark_merge,
        df1=df1, df2=df2, how=how, sort=sort
    )
    return result
