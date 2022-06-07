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
import pandas
import numpy as np
import pyarrow
import pytest
import re

from modin.config import StorageFormat
from modin.pandas.test.utils import io_ops_bad_exc, default_to_pandas_ignore_string
from .utils import eval_io, ForceOmnisciImport, set_execution_mode, run_and_compare
from pandas.core.dtypes.common import is_list_like

StorageFormat.put("omnisci")

import modin.pandas as pd
from modin.pandas.test.utils import (
    df_equals,
    bool_arg_values,
    to_pandas,
    test_data_values,
    test_data_keys,
    generate_multiindex,
    eval_general,
    df_equals_with_non_stable_indices,
    time_parsing_csv_path,
)
from modin.utils import try_cast_to_pandas
from modin.pandas.utils import from_arrow

from modin.experimental.core.execution.native.implementations.omnisci_on_native.partitioning.partition_manager import (
    OmnisciOnNativeDataframePartitionManager,
)
from modin.experimental.core.execution.native.implementations.omnisci_on_native.df_algebra import (
    FrameNode,
)


# Our configuration in pytest.ini requires that we explicitly catch all
# instances of defaulting to pandas, but some test modules, like this one,
# have too many such instances.
# TODO(https://github.com/modin-project/modin/issues/3655): catch all instances
# of defaulting to pandas.
pytestmark = pytest.mark.filterwarnings(default_to_pandas_ignore_string)


@pytest.mark.usefixtures("TestReadCSVFixture")
class TestCSV:
    from modin import __file__ as modin_root

    root = os.path.dirname(
        os.path.dirname(os.path.abspath(modin_root)) + ".."
    )  # root of modin repo

    boston_housing_names = [
        "index",
        "CRIM",
        "ZN",
        "INDUS",
        "CHAS",
        "NOX",
        "RM",
        "AGE",
        "DIS",
        "RAD",
        "TAX",
        "PTRATIO",
        "B",
        "LSTAT",
        "PRICE",
    ]
    boston_housing_dtypes = {
        "index": "int64",
        "CRIM": "float64",
        "ZN": "float64",
        "INDUS": "float64",
        "CHAS": "float64",
        "NOX": "float64",
        "RM": "float64",
        "AGE": "float64",
        "DIS": "float64",
        "RAD": "float64",
        "TAX": "float64",
        "PTRATIO": "float64",
        "B": "float64",
        "LSTAT": "float64",
        "PRICE": "float64",
    }

    def test_usecols_csv(self):
        """check with the following arguments: names, dtype, skiprows, delimiter"""
        csv_file = os.path.join(self.root, "modin/pandas/test/data", "test_usecols.csv")

        for kwargs in (
            {"delimiter": ","},
            {"sep": None},
            {"skiprows": 1, "names": ["A", "B", "C", "D", "E"]},
            {"dtype": {"a": "int32", "e": "string"}},
            {"dtype": {"a": np.dtype("int32"), "b": np.dtype("int64"), "e": "string"}},
        ):
            eval_io(
                fn_name="read_csv",
                md_extra_kwargs={"engine": "arrow"},
                # read_csv kwargs
                filepath_or_buffer=csv_file,
                **kwargs,
            )

    def test_housing_csv(self):
        csv_file = os.path.join(self.root, "examples/data/boston_housing.csv")
        for kwargs in (
            {
                "skiprows": 1,
                "names": self.boston_housing_names,
                "dtype": self.boston_housing_dtypes,
            },
        ):
            eval_io(
                fn_name="read_csv",
                md_extra_kwargs={"engine": "arrow"},
                # read_csv kwargs
                filepath_or_buffer=csv_file,
                **kwargs,
            )

    def test_time_parsing(self):
        csv_file = os.path.join(self.root, time_parsing_csv_path)
        for kwargs in (
            {
                "skiprows": 1,
                "names": [
                    "timestamp",
                    "year",
                    "month",
                    "date",
                    "symbol",
                    "high",
                    "low",
                    "open",
                    "close",
                    "spread",
                    "volume",
                ],
                "parse_dates": ["timestamp"],
                "dtype": {"symbol": "string"},
            },
        ):
            rp = pandas.read_csv(csv_file, **kwargs)
            rm = pd.read_csv(csv_file, engine="arrow", **kwargs)
            with ForceOmnisciImport(rm):
                rm = to_pandas(rm)
                df_equals(rm["timestamp"].dt.year, rp["timestamp"].dt.year)
                df_equals(rm["timestamp"].dt.month, rp["timestamp"].dt.month)
                df_equals(rm["timestamp"].dt.day, rp["timestamp"].dt.day)
                df_equals(rm["timestamp"].dt.hour, rp["timestamp"].dt.hour)

    def test_csv_fillna(self):
        csv_file = os.path.join(self.root, "examples/data/boston_housing.csv")
        for kwargs in (
            {
                "skiprows": 1,
                "names": self.boston_housing_names,
                "dtype": self.boston_housing_dtypes,
            },
        ):
            eval_io(
                fn_name="read_csv",
                md_extra_kwargs={"engine": "arrow"},
                comparator=lambda df1, df2: df_equals(
                    df1["CRIM"].fillna(1000), df2["CRIM"].fillna(1000)
                ),
                # read_csv kwargs
                filepath_or_buffer=csv_file,
                **kwargs,
            )

    @pytest.mark.parametrize("null_dtype", ["category", "float64"])
    def test_null_col(self, null_dtype):
        csv_file = os.path.join(
            self.root, "modin/pandas/test/data", "test_null_col.csv"
        )
        ref = pandas.read_csv(
            csv_file,
            names=["a", "b", "c"],
            dtype={"a": "int64", "b": "int64", "c": null_dtype},
            skiprows=1,
        )
        ref["a"] = ref["a"] + ref["b"]

        exp = pd.read_csv(
            csv_file,
            names=["a", "b", "c"],
            dtype={"a": "int64", "b": "int64", "c": null_dtype},
            skiprows=1,
        )
        exp["a"] = exp["a"] + exp["b"]

        # df_equals cannot compare empty categories
        if null_dtype == "category":
            ref["c"] = ref["c"].astype("string")
            with ForceOmnisciImport(exp):
                exp = to_pandas(exp)
            exp["c"] = exp["c"].astype("string")

        df_equals(ref, exp)

    def test_read_and_concat(self):
        csv_file = os.path.join(self.root, "modin/pandas/test/data", "test_usecols.csv")
        ref1 = pandas.read_csv(csv_file)
        ref2 = pandas.read_csv(csv_file)
        ref = pandas.concat([ref1, ref2])

        exp1 = pandas.read_csv(csv_file)
        exp2 = pandas.read_csv(csv_file)
        exp = pd.concat([exp1, exp2])
        with ForceOmnisciImport(exp):
            df_equals(ref, exp)

    @pytest.mark.parametrize("names", [None, ["a", "b", "c", "d", "e"]])
    @pytest.mark.parametrize("header", [None, 0])
    def test_from_csv(self, header, names):
        csv_file = os.path.join(self.root, "modin/pandas/test/data", "test_usecols.csv")
        eval_io(
            fn_name="read_csv",
            # read_csv kwargs
            filepath_or_buffer=csv_file,
            header=header,
            names=names,
        )

    @pytest.mark.parametrize("kwargs", [{"sep": "|"}, {"delimiter": "|"}])
    def test_sep_delimiter(self, kwargs):
        csv_file = os.path.join(self.root, "modin/pandas/test/data", "test_delim.csv")
        eval_io(
            fn_name="read_csv",
            # read_csv kwargs
            filepath_or_buffer=csv_file,
            **kwargs,
        )

    @pytest.mark.skip(reason="https://github.com/modin-project/modin/issues/2174")
    def test_float32(self):
        csv_file = os.path.join(self.root, "modin/pandas/test/data", "test_usecols.csv")
        kwargs = {
            "dtype": {"a": "float32", "b": "float32"},
        }

        pandas_df = pandas.read_csv(csv_file, **kwargs)
        pandas_df["a"] = pandas_df["a"] + pandas_df["b"]

        modin_df = pd.read_csv(csv_file, **kwargs, engine="arrow")
        modin_df["a"] = modin_df["a"] + modin_df["b"]
        with ForceOmnisciImport(modin_df):
            df_equals(modin_df, pandas_df)

    # Datetime Handling tests
    @pytest.mark.parametrize("engine", [None, "arrow"])
    @pytest.mark.parametrize(
        "parse_dates",
        [
            True,
            False,
            ["col2"],
            ["c2"],
            [["col2", "col3"]],
            {"col23": ["col2", "col3"]},
            [],
        ],
    )
    @pytest.mark.parametrize("names", [None, [f"c{x}" for x in range(1, 7)]])
    def test_read_csv_datetime(
        self,
        engine,
        parse_dates,
        names,
    ):

        parse_dates_unsupported = isinstance(parse_dates, dict) or (
            isinstance(parse_dates, list)
            and any(not isinstance(date, str) for date in parse_dates)
        )
        if parse_dates_unsupported and engine == "arrow" and not names:
            pytest.skip(
                "In these cases Modin raises `ArrowEngineException` while pandas "
                + "doesn't raise any exceptions that causes tests fails"
            )
        # In these cases Modin raises `ArrowEngineException` while pandas
        # raises `ValueError`, so skipping exception type checking
        skip_exc_type_check = parse_dates_unsupported and engine == "arrow"

        eval_io(
            fn_name="read_csv",
            md_extra_kwargs={"engine": engine},
            check_exception_type=not skip_exc_type_check,
            raising_exceptions=None if skip_exc_type_check else io_ops_bad_exc,
            # read_csv kwargs
            filepath_or_buffer=pytest.csvs_names["test_read_csv_regular"],
            parse_dates=parse_dates,
            names=names,
        )

    @pytest.mark.parametrize("engine", [None, "arrow"])
    @pytest.mark.parametrize(
        "usecols",
        [
            None,
            ["col1"],
            ["col1", "col1"],
            ["col1", "col2", "col6"],
            ["col6", "col2", "col1"],
            [0],
            [0, 0],
            [0, 1, 5],
            [5, 1, 0],
            lambda x: x in ["col1", "col2"],
        ],
    )
    def test_read_csv_col_handling(
        self,
        engine,
        usecols,
    ):
        eval_io(
            fn_name="read_csv",
            check_kwargs_callable=not callable(usecols),
            md_extra_kwargs={"engine": engine},
            # read_csv kwargs
            filepath_or_buffer=pytest.csvs_names["test_read_csv_regular"],
            usecols=usecols,
        )


class TestMasks:
    data = {
        "a": [1, 1, 2, 2, 3],
        "b": [None, None, 2, 1, 3],
        "c": [3, None, None, 2, 1],
    }
    cols_values = ["a", ["a", "b"], ["a", "b", "c"]]

    @pytest.mark.parametrize("cols", cols_values)
    def test_projection(self, cols):
        def projection(df, cols, **kwargs):
            return df[cols]

        run_and_compare(projection, data=self.data, cols=cols)

    def test_drop(self):
        def drop(df, **kwargs):
            return df.drop(columns="a")

        run_and_compare(drop, data=self.data)

    def test_iloc(self):
        def mask(df, **kwargs):
            return df.iloc[[0, 1]]

        run_and_compare(mask, data=self.data, allow_subqueries=True)

    def test_empty(self):
        def empty(df, **kwargs):
            return df

        run_and_compare(empty, data=None)

    def test_filter(self):
        def filter(df, **kwargs):
            return df[df["a"] == 1]

        run_and_compare(filter, data=self.data)

    def test_filter_with_index(self):
        def filter(df, **kwargs):
            df = df.groupby("a").sum()
            return df[df["b"] > 1]

        run_and_compare(filter, data=self.data)

    def test_filter_proj(self):
        def filter(df, **kwargs):
            df1 = df + 2
            return df1[(df["a"] + df1["b"]) > 1]

        run_and_compare(filter, data=self.data)

    def test_filter_drop(self):
        def filter(df, **kwargs):
            df = df[["a", "b"]]
            df = df[df["a"] != 1]
            df["a"] = df["a"] * df["b"]
            return df

        run_and_compare(filter, data=self.data)


class TestMultiIndex:
    data = {"a": np.arange(24), "b": np.arange(24)}

    @pytest.mark.parametrize("names", [None, ["", ""], ["name", "name"]])
    def test_dup_names(self, names):
        index = pandas.MultiIndex.from_tuples(
            [(i, j) for i in range(3) for j in range(8)], names=names
        )

        pandas_df = pandas.DataFrame(self.data, index=index) + 1
        modin_df = pd.DataFrame(self.data, index=index) + 1

        df_equals(pandas_df, modin_df)

    @pytest.mark.parametrize(
        "names",
        [
            None,
            [None, "s", None],
            ["i1", "i2", "i3"],
            ["i1", "i1", "i3"],
            ["i1", "i2", "a"],
        ],
    )
    def test_reset_index(self, names):
        index = pandas.MultiIndex.from_tuples(
            [(i, j, k) for i in range(2) for j in range(3) for k in range(4)],
            names=names,
        )

        def applier(lib):
            df = lib.DataFrame(self.data, index=index) + 1
            return df.reset_index()

        eval_general(pd, pandas, applier)

    @pytest.mark.parametrize("is_multiindex", [True, False])
    @pytest.mark.parametrize(
        "column_names", [None, ["level1", None], ["level1", "level2"]]
    )
    def test_reset_index_multicolumns(self, is_multiindex, column_names):
        index = (
            pandas.MultiIndex.from_tuples(
                [(i, j, k) for i in range(2) for j in range(3) for k in range(4)],
                names=["l1", "l2", "l3"],
            )
            if is_multiindex
            else pandas.Index(np.arange(len(self.data["a"])), name="index")
        )
        columns = pandas.MultiIndex.from_tuples(
            [("a", "b"), ("b", "c")], names=column_names
        )
        data = np.array(list(self.data.values())).T

        def applier(df, **kwargs):
            df = df + 1
            return df.reset_index(drop=False)

        run_and_compare(
            fn=applier,
            data=data,
            constructor_kwargs={"index": index, "columns": columns},
        )

    def test_set_index_name(self):
        index = pandas.Index.__new__(pandas.Index, data=[i for i in range(24)])

        pandas_df = pandas.DataFrame(self.data, index=index)
        pandas_df.index.name = "new_name"
        modin_df = pd.DataFrame(self.data, index=index)
        modin_df._query_compiler.set_index_name("new_name")

        df_equals(pandas_df, modin_df)

    def test_set_index_names(self):
        index = pandas.MultiIndex.from_tuples(
            [(i, j, k) for i in range(2) for j in range(3) for k in range(4)]
        )

        pandas_df = pandas.DataFrame(self.data, index=index)
        pandas_df.index.names = ["new_name1", "new_name2", "new_name3"]
        modin_df = pd.DataFrame(self.data, index=index)
        modin_df._query_compiler.set_index_names(
            ["new_name1", "new_name2", "new_name3"]
        )

        df_equals(pandas_df, modin_df)


class TestFillna:
    data = {"a": [1, 1, None], "b": [None, None, 2], "c": [3, None, None]}
    values = [1, {"a": 1, "c": 3}, {"a": 1, "b": 2, "c": 3}]

    @pytest.mark.parametrize("value", values)
    def test_fillna_all(self, value):
        def fillna(df, value, **kwargs):
            return df.fillna(value)

        run_and_compare(fillna, data=self.data, value=value)

    def test_fillna_bool(self):
        def fillna(df, **kwargs):
            df["a"] = df["a"] == 1
            df["a"] = df["a"].fillna(False)
            return df

        run_and_compare(fillna, data=self.data)


class TestConcat:
    data = {
        "a": [1, 2, 3],
        "b": [10, 20, 30],
        "d": [1000, 2000, 3000],
        "e": [11, 22, 33],
    }
    data2 = {
        "a": [4, 5, 6],
        "c": [400, 500, 600],
        "b": [40, 50, 60],
        "f": [444, 555, 666],
    }
    data3 = {
        "f": [2, 3, 4],
        "g": [400, 500, 600],
        "h": [20, 30, 40],
    }

    @pytest.mark.parametrize("join", ["inner", "outer"])
    @pytest.mark.parametrize("sort", bool_arg_values)
    @pytest.mark.parametrize("ignore_index", bool_arg_values)
    def test_concat(self, join, sort, ignore_index):
        def concat(lib, df1, df2, join, sort, ignore_index):
            return lib.concat(
                [df1, df2], join=join, sort=sort, ignore_index=ignore_index
            )

        run_and_compare(
            concat,
            data=self.data,
            data2=self.data2,
            join=join,
            sort=sort,
            ignore_index=ignore_index,
        )

    def test_concat_with_same_df(self):
        def concat(df, **kwargs):
            df["f"] = df["a"]
            return df

        run_and_compare(concat, data=self.data)

    def test_setitem_lazy(self):
        def applier(df, **kwargs):
            df = df + 1
            df["a"] = df["a"] + 1
            df["e"] = df["a"] + 1
            df["new_int8"] = np.int8(10)
            df["new_int16"] = np.int16(10)
            df["new_int32"] = np.int32(10)
            df["new_int64"] = np.int64(10)
            df["new_int"] = 10
            df["new_float"] = 5.5
            df["new_float64"] = np.float64(10.1)
            return df

        run_and_compare(applier, data=self.data)

    def test_setitem_default(self):
        def applier(df, lib, **kwargs):
            df = df + 1
            df["a"] = np.arange(3)
            df["b"] = lib.Series(np.arange(3))
            return df

        run_and_compare(applier, data=self.data, force_lazy=False)

    def test_insert_lazy(self):
        def applier(df, **kwargs):
            df = df + 1
            df.insert(2, "new_int", 10)
            df.insert(1, "new_float", 5.5)
            df.insert(0, "new_a", df["a"] + 1)
            return df

        run_and_compare(applier, data=self.data)

    def test_insert_default(self):
        def applier(df, lib, **kwargs):
            df = df + 1
            df.insert(1, "new_range", np.arange(3))
            df.insert(1, "new_series", lib.Series(np.arange(3)))
            return df

        run_and_compare(applier, data=self.data, force_lazy=False)

    def test_concat_many(self):
        def concat(df1, df2, lib, **kwargs):
            df3 = df1.copy()
            df4 = df2.copy()
            return lib.concat([df1, df2, df3, df4])

        def sort_comparator(df1, df2):
            """Sort and verify equality of the passed frames."""
            # We sort values because order of rows in the 'union all' result is inconsistent in OmniSci
            df1, df2 = (
                try_cast_to_pandas(df).sort_values(df.columns[0]) for df in (df1, df2)
            )
            return df_equals(df1, df2)

        run_and_compare(
            concat, data=self.data, data2=self.data2, comparator=sort_comparator
        )

    def test_concat_agg(self):
        def concat(lib, df1, df2):
            df1 = df1.groupby("a", as_index=False).agg(
                {"b": "sum", "d": "sum", "e": "sum"}
            )
            df2 = df2.groupby("a", as_index=False).agg(
                {"c": "sum", "b": "sum", "f": "sum"}
            )
            return lib.concat([df1, df2])

        run_and_compare(concat, data=self.data, data2=self.data2, allow_subqueries=True)

    @pytest.mark.parametrize("join", ["inner", "outer"])
    @pytest.mark.parametrize("sort", bool_arg_values)
    @pytest.mark.parametrize("ignore_index", bool_arg_values)
    def test_concat_single(self, join, sort, ignore_index):
        def concat(lib, df, join, sort, ignore_index):
            return lib.concat([df], join=join, sort=sort, ignore_index=ignore_index)

        run_and_compare(
            concat,
            data=self.data,
            join=join,
            sort=sort,
            ignore_index=ignore_index,
        )

    def test_groupby_concat_single(self):
        def concat(lib, df):
            df = lib.concat([df])
            return df.groupby("a").agg({"b": "min"})

        run_and_compare(
            concat,
            data=self.data,
        )

    @pytest.mark.parametrize("join", ["inner"])
    @pytest.mark.parametrize("sort", bool_arg_values)
    @pytest.mark.parametrize("ignore_index", bool_arg_values)
    def test_concat_join(self, join, sort, ignore_index):
        def concat(lib, df1, df2, join, sort, ignore_index, **kwargs):
            return lib.concat(
                [df1, df2], axis=1, join=join, sort=sort, ignore_index=ignore_index
            )

        run_and_compare(
            concat,
            data=self.data,
            data2=self.data3,
            join=join,
            sort=sort,
            ignore_index=ignore_index,
        )

    def test_concat_index_name(self):
        df1 = pandas.DataFrame(self.data)
        df1 = df1.set_index("a")
        df2 = pandas.DataFrame(self.data3)
        df2 = df2.set_index("f")

        ref = pandas.concat([df1, df2], axis=1, join="inner")
        exp = pd.concat([df1, df2], axis=1, join="inner")

        df_equals(ref, exp)

        df2.index.name = "a"
        ref = pandas.concat([df1, df2], axis=1, join="inner")
        exp = pd.concat([df1, df2], axis=1, join="inner")

        df_equals(ref, exp)

    def test_concat_index_names(self):
        df1 = pandas.DataFrame(self.data)
        df1 = df1.set_index(["a", "b"])
        df2 = pandas.DataFrame(self.data3)
        df2 = df2.set_index(["f", "h"])

        ref = pandas.concat([df1, df2], axis=1, join="inner")
        exp = pd.concat([df1, df2], axis=1, join="inner")

        df_equals(ref, exp)

        df2.index.names = ["a", "b"]
        ref = pandas.concat([df1, df2], axis=1, join="inner")
        exp = pd.concat([df1, df2], axis=1, join="inner")

        df_equals(ref, exp)


class TestGroupby:
    data = {
        "a": [1, 1, 2, 2, 2, 1],
        "b": [11, 21, 12, 22, 32, 11],
        "c": [101, 201, 202, 202, 302, 302],
    }
    cols_value = ["a", ["a", "b"]]

    @pytest.mark.parametrize("cols", cols_value)
    @pytest.mark.parametrize("as_index", bool_arg_values)
    def test_groupby_sum(self, cols, as_index):
        def groupby_sum(df, cols, as_index, **kwargs):
            return df.groupby(cols, as_index=as_index).sum()

        run_and_compare(groupby_sum, data=self.data, cols=cols, as_index=as_index)

    @pytest.mark.parametrize("cols", cols_value)
    @pytest.mark.parametrize("as_index", bool_arg_values)
    def test_groupby_count(self, cols, as_index):
        def groupby_count(df, cols, as_index, **kwargs):
            return df.groupby(cols, as_index=as_index).count()

        run_and_compare(groupby_count, data=self.data, cols=cols, as_index=as_index)

    @pytest.mark.parametrize("cols", cols_value)
    @pytest.mark.parametrize("as_index", bool_arg_values)
    def test_groupby_mean(self, cols, as_index):
        def groupby_mean(df, cols, as_index, **kwargs):
            return df.groupby(cols, as_index=as_index).mean()

        run_and_compare(groupby_mean, data=self.data, cols=cols, as_index=as_index)

    @pytest.mark.parametrize("cols", cols_value)
    @pytest.mark.parametrize("as_index", bool_arg_values)
    def test_groupby_proj_sum(self, cols, as_index):
        def groupby_sum(df, cols, as_index, **kwargs):
            return df.groupby(cols, as_index=as_index).c.sum()

        run_and_compare(
            groupby_sum, data=self.data, cols=cols, as_index=as_index, force_lazy=False
        )

    @pytest.mark.parametrize("agg", ["count", "size", "nunique"])
    def test_groupby_agg(self, agg):
        def groupby(df, agg, **kwargs):
            return df.groupby("a").agg({"b": agg})

        run_and_compare(groupby, data=self.data, agg=agg)

    def test_groupby_agg_default_to_pandas(self):
        def lambda_func(df, **kwargs):
            return df.groupby("a").agg(lambda df: (df.mean() - df.sum()) // 2)

        run_and_compare(lambda_func, data=self.data, force_lazy=False)

        def not_implemented_func(df, **kwargs):
            return df.groupby("a").agg("cumprod")

        run_and_compare(lambda_func, data=self.data, force_lazy=False)

    @pytest.mark.parametrize("cols", cols_value)
    @pytest.mark.parametrize("as_index", bool_arg_values)
    def test_groupby_agg_mean(self, cols, as_index):
        def groupby_mean(df, cols, as_index, **kwargs):
            return df.groupby(cols, as_index=as_index).agg("mean")

        run_and_compare(groupby_mean, data=self.data, cols=cols, as_index=as_index)

    def test_groupby_lazy_multiindex(self):
        index = generate_multiindex(len(self.data["a"]))

        def groupby(df, *args, **kwargs):
            df = df + 1
            return df.groupby("a").agg({"b": "size"})

        run_and_compare(groupby, data=self.data, constructor_kwargs={"index": index})

    def test_groupby_lazy_squeeze(self):
        def applier(df, **kwargs):
            return df.groupby("a").sum().squeeze(axis=1)

        run_and_compare(
            applier,
            data=self.data,
            constructor_kwargs={"columns": ["a", "b"]},
            force_lazy=True,
        )

    @pytest.mark.parametrize("method", ["sum", "size"])
    def test_groupby_series(self, method):
        def groupby(df, **kwargs):
            ser = df[df.columns[0]]
            return getattr(ser.groupby(ser), method)()

        run_and_compare(groupby, data=self.data)

    def test_groupby_size(self):
        def groupby(df, **kwargs):
            return df.groupby("a").size()

        run_and_compare(groupby, data=self.data)

    @pytest.mark.parametrize("by", [["a"], ["a", "b", "c"]])
    @pytest.mark.parametrize("agg", ["sum", "size", "mean"])
    @pytest.mark.parametrize("as_index", [True, False])
    def test_groupby_agg_by_col(self, by, agg, as_index):
        def simple_agg(df, **kwargs):
            return df.groupby(by, as_index=as_index).agg(agg)

        run_and_compare(simple_agg, data=self.data)

        def dict_agg(df, **kwargs):
            return df.groupby(by, as_index=as_index).agg({by[0]: agg})

        run_and_compare(dict_agg, data=self.data)

        def dict_agg_all_cols(df, **kwargs):
            return df.groupby(by, as_index=as_index).agg({col: agg for col in by})

        run_and_compare(dict_agg_all_cols, data=self.data)

    # modin-issue#3461
    def test_groupby_pure_by(self):
        data = [1, 1, 2, 2]
        # Test when 'by' is a 'TransformNode'
        run_and_compare(lambda df: df.groupby(df).sum(), data=data, force_lazy=True)

        # Test when 'by' is a 'FrameNode'
        md_ser, pd_ser = pd.Series(data), pandas.Series(data)

        md_ser._query_compiler._modin_frame._execute()
        assert isinstance(
            md_ser._query_compiler._modin_frame._op, FrameNode
        ), "Triggering execution of the Modin frame supposed to set 'FrameNode' as a frame's op"

        set_execution_mode(md_ser, "lazy")
        md_res = md_ser.groupby(md_ser).sum()
        set_execution_mode(md_res, None)

        pd_res = pd_ser.groupby(pd_ser).sum()
        df_equals(md_res, pd_res)

    taxi_data = {
        "a": [1, 1, 2, 2],
        "b": [11, 21, 12, 11],
        "c": pandas.to_datetime(
            ["20190902", "20180913", "20190921", "20180903"], format="%Y%m%d"
        ),
        "d": [11.5, 21.2, 12.8, 13.4],
    }

    # TODO: emulate taxi queries with group by category types when we have loading
    #       using arrow
    #       Another way of doing taxi q1 is
    #       res = df.groupby("cab_type").size() - this should be tested later as well
    def test_taxi_q1(self):
        def taxi_q1(df, **kwargs):
            return df.groupby("a").size()

        run_and_compare(taxi_q1, data=self.taxi_data)

    def test_taxi_q2(self):
        def taxi_q2(df, **kwargs):
            return df.groupby("a").agg({"b": "mean"})

        run_and_compare(taxi_q2, data=self.taxi_data)

    @pytest.mark.parametrize("as_index", bool_arg_values)
    def test_taxi_q3(self, as_index):
        def taxi_q3(df, as_index, **kwargs):
            return df.groupby(["b", df["c"].dt.year], as_index=as_index).size()

        run_and_compare(taxi_q3, data=self.taxi_data, as_index=as_index)

    def test_groupby_expr_col(self):
        def groupby(df, **kwargs):
            df = df.loc[:, ["b", "c"]]
            df["year"] = df["c"].dt.year
            df["month"] = df["c"].dt.month
            df["id1"] = df["year"] * 12 + df["month"]
            df["id2"] = (df["id1"] - 24000) // 12
            df = df.groupby(["id1", "id2"], as_index=False).agg({"b": "max"})
            return df

        run_and_compare(groupby, data=self.taxi_data)

    def test_series_astype(self):
        def series_astype(df, **kwargs):
            return df["d"].astype("int")

        run_and_compare(series_astype, data=self.taxi_data)

    def test_df_astype(self):
        def df_astype(df, **kwargs):
            return df.astype({"b": "float", "d": "int"})

        run_and_compare(df_astype, data=self.taxi_data)

    def test_df_indexed_astype(self):
        def df_astype(df, **kwargs):
            df = df.groupby("a").agg({"b": "sum"})
            return df.astype({"b": "float"})

        run_and_compare(df_astype, data=self.taxi_data)

    @pytest.mark.parametrize("as_index", bool_arg_values)
    def test_taxi_q4(self, as_index):
        def taxi_q4(df, **kwargs):
            df["c"] = df["c"].dt.year
            df["d"] = df["d"].astype("int64")
            df = df.groupby(["b", "c", "d"], sort=True, as_index=as_index).size()
            if as_index:
                df = df.reset_index()
            return df.sort_values(
                by=["c", 0 if as_index else "size"],
                ignore_index=True,
                ascending=[True, False],
            )

        run_and_compare(taxi_q4, data=self.taxi_data)

    h2o_data = {
        "id1": ["id1", "id2", "id3", "id1", "id2", "id3", "id1", "id2", "id3", "id1"],
        "id2": ["id1", "id2", "id1", "id2", "id1", "id2", "id1", "id2", "id1", "id2"],
        "id3": ["id4", "id5", "id6", "id4", "id5", "id6", "id4", "id5", "id6", "id4"],
        "id4": [4, 5, 4, 5, 4, 5, 4, 5, 4, 5],
        "id5": [7, 8, 9, 7, 8, 9, 7, 8, 9, 7],
        "id6": [7, 8, 7, 8, 7, 8, 7, 8, 7, 8],
        "v1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "v2": [1, 3, 5, 7, 9, 10, 8, 6, 4, 2],
        "v3": [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.0],
    }

    def _get_h2o_df(self):
        df = pandas.DataFrame(self.h2o_data)
        df["id1"] = df["id1"].astype("category")
        df["id2"] = df["id2"].astype("category")
        df["id3"] = df["id3"].astype("category")
        return df

    def test_h2o_q1(self):
        df = self._get_h2o_df()

        ref = df.groupby(["id1"], observed=True).agg({"v1": "sum"})
        ref.reset_index(inplace=True)

        modin_df = pd.DataFrame(df)
        set_execution_mode(modin_df, "lazy")
        modin_df = modin_df.groupby(["id1"], observed=True, as_index=False).agg(
            {"v1": "sum"}
        )
        set_execution_mode(modin_df, None)

        exp = to_pandas(modin_df)
        exp["id1"] = exp["id1"].astype("category")

        df_equals(ref, exp)

    def test_h2o_q2(self):
        df = self._get_h2o_df()

        ref = df.groupby(["id1", "id2"], observed=True).agg({"v1": "sum"})
        ref.reset_index(inplace=True)

        modin_df = pd.DataFrame(df)
        set_execution_mode(modin_df, "lazy")
        modin_df = modin_df.groupby(["id1", "id2"], observed=True, as_index=False).agg(
            {"v1": "sum"}
        )
        set_execution_mode(modin_df, None)

        exp = to_pandas(modin_df)
        exp["id1"] = exp["id1"].astype("category")
        exp["id2"] = exp["id2"].astype("category")

        df_equals(ref, exp)

    def test_h2o_q3(self):
        df = self._get_h2o_df()

        ref = df.groupby(["id3"], observed=True).agg({"v1": "sum", "v3": "mean"})
        ref.reset_index(inplace=True)

        modin_df = pd.DataFrame(df)
        set_execution_mode(modin_df, "lazy")
        modin_df = modin_df.groupby(["id3"], observed=True, as_index=False).agg(
            {"v1": "sum", "v3": "mean"}
        )
        set_execution_mode(modin_df, None)

        exp = to_pandas(modin_df)
        exp["id3"] = exp["id3"].astype("category")

        df_equals(ref, exp)

    def test_h2o_q4(self):
        df = self._get_h2o_df()

        ref = df.groupby(["id4"], observed=True).agg(
            {"v1": "mean", "v2": "mean", "v3": "mean"}
        )
        ref.reset_index(inplace=True)

        modin_df = pd.DataFrame(df)
        set_execution_mode(modin_df, "lazy")
        modin_df = modin_df.groupby(["id4"], observed=True, as_index=False).agg(
            {"v1": "mean", "v2": "mean", "v3": "mean"}
        )
        set_execution_mode(modin_df, None)

        exp = to_pandas(modin_df)

        df_equals(ref, exp)

    def test_h2o_q5(self):
        df = self._get_h2o_df()

        ref = df.groupby(["id6"], observed=True).agg(
            {"v1": "sum", "v2": "sum", "v3": "sum"}
        )
        ref.reset_index(inplace=True)

        modin_df = pd.DataFrame(df)
        set_execution_mode(modin_df, "lazy")
        modin_df = modin_df.groupby(["id6"], observed=True, as_index=False).agg(
            {"v1": "sum", "v2": "sum", "v3": "sum"}
        )
        set_execution_mode(modin_df, None)

        exp = to_pandas(modin_df)

        df_equals(ref, exp)

    def test_h2o_q7(self):
        df = self._get_h2o_df()

        ref = (
            df.groupby(["id3"], observed=True)
            .agg({"v1": "max", "v2": "min"})
            .assign(range_v1_v2=lambda x: x["v1"] - x["v2"])[["range_v1_v2"]]
        )
        ref.reset_index(inplace=True)

        modin_df = pd.DataFrame(df)
        set_execution_mode(modin_df, "lazy")
        modin_df = modin_df.groupby(["id3"], observed=True).agg(
            {"v1": "max", "v2": "min"}
        )
        modin_df["range_v1_v2"] = modin_df["v1"] - modin_df["v2"]
        modin_df = modin_df[["range_v1_v2"]]
        modin_df.reset_index(inplace=True)
        set_execution_mode(modin_df, None)

        exp = to_pandas(modin_df)
        exp["id3"] = exp["id3"].astype("category")

        df_equals(ref, exp)

    def test_h2o_q10(self):
        df = self._get_h2o_df()

        ref = df.groupby(["id1", "id2", "id3", "id4", "id5", "id6"], observed=True).agg(
            {"v3": "sum", "v1": "count"}
        )
        ref.reset_index(inplace=True)

        modin_df = pd.DataFrame(df)
        modin_df = modin_df.groupby(
            ["id1", "id2", "id3", "id4", "id5", "id6"], observed=True
        ).agg({"v3": "sum", "v1": "count"})
        modin_df.reset_index(inplace=True)

        exp = to_pandas(modin_df)
        exp["id1"] = exp["id1"].astype("category")
        exp["id2"] = exp["id2"].astype("category")
        exp["id3"] = exp["id3"].astype("category")

        df_equals(ref, exp)

    std_data = {
        "a": [1, 2, 1, 1, 1, 2, 2, 2, 1, 2],
        "b": [4, 3, 1, 6, 9, 8, 0, 9, 5, 13],
        "c": [12.8, 45.6, 23.5, 12.4, 11.2, None, 56.4, 12.5, 1, 55],
    }

    def test_agg_std(self):
        def std(df, **kwargs):
            df = df.groupby("a").agg({"b": "std", "c": "std"})
            if not isinstance(df, pandas.DataFrame):
                df = to_pandas(df)
            df["b"] = df["b"].apply(lambda x: round(x, 10))
            df["c"] = df["c"].apply(lambda x: round(x, 10))
            return df

        run_and_compare(std, data=self.std_data, force_lazy=False)

    skew_data = {
        "a": [1, 2, 1, 1, 1, 2, 2, 2, 1, 2, 3, 4, 4],
        "b": [4, 3, 1, 6, 9, 8, 0, 9, 5, 13, 12, 44, 6],
        "c": [12.8, 45.6, 23.5, 12.4, 11.2, None, 56.4, 12.5, 1, 55, 4.5, 7.8, 9.4],
    }

    def test_agg_skew(self):
        def std(df, **kwargs):
            df = df.groupby("a").agg({"b": "skew", "c": "skew"})
            if not isinstance(df, pandas.DataFrame):
                df = to_pandas(df)
            df["b"] = df["b"].apply(lambda x: round(x, 10))
            df["c"] = df["c"].apply(lambda x: round(x, 10))
            return df

        run_and_compare(std, data=self.skew_data, force_lazy=False)

    def test_multilevel(self):
        def groupby(df, **kwargs):
            return df.groupby("a").agg({"b": "min", "c": ["min", "max", "sum", "skew"]})

        run_and_compare(groupby, data=self.data)


class TestAgg:
    data = {
        "a": [1, 2, None, None, 1, None],
        "b": [10, 20, None, 20, 10, None],
        "c": [None, 200, None, 400, 500, 600],
        "d": [11, 22, 33, 22, 33, 22],
    }
    int_data = pandas.DataFrame(data).fillna(0).astype("int").to_dict()

    @pytest.mark.parametrize("agg", ["max", "min", "sum", "mean"])
    @pytest.mark.parametrize("skipna", bool_arg_values)
    def test_simple_agg(self, agg, skipna):
        def apply(df, agg, skipna, **kwargs):
            return getattr(df, agg)(skipna=skipna)

        run_and_compare(apply, data=self.data, agg=agg, skipna=skipna, force_lazy=False)

    def test_count_agg(self):
        def apply(df, **kwargs):
            return df.count()

        run_and_compare(apply, data=self.data, force_lazy=False)

    @pytest.mark.parametrize("data", [data, int_data], ids=["nan_data", "int_data"])
    @pytest.mark.parametrize("cols", ["a", "d", ["a", "d"]])
    @pytest.mark.parametrize("dropna", [True, False])
    @pytest.mark.parametrize("sort", [True])
    @pytest.mark.parametrize("ascending", [True, False])
    def test_value_counts(self, data, cols, dropna, sort, ascending):
        def value_counts(df, cols, dropna, sort, ascending, **kwargs):
            return df[cols].value_counts(dropna=dropna, sort=sort, ascending=ascending)

        if dropna and pandas.DataFrame(
            data, columns=cols if is_list_like(cols) else [cols]
        ).isna().any(axis=None):
            pytest.xfail(
                reason="'dropna' parameter is forcibly disabled in OmniSci's GroupBy"
                + "due to performance issues, you can track this problem at:"
                + "https://github.com/modin-project/modin/issues/2896"
            )

        # Custom comparator is required because pandas is inconsistent about
        # the order of equal values, we can't match this behavior. For more details:
        # https://github.com/modin-project/modin/issues/1650
        run_and_compare(
            value_counts,
            data=data,
            cols=cols,
            dropna=dropna,
            sort=sort,
            ascending=ascending,
            comparator=df_equals_with_non_stable_indices,
        )

    @pytest.mark.parametrize(
        "method", ["sum", "mean", "max", "min", "count", "nunique"]
    )
    def test_simple_agg_no_default(self, method):
        def applier(df, **kwargs):
            if isinstance(df, pd.DataFrame):
                # At the end of reduce function it does inevitable `transpose`, which
                # is defaulting to pandas. The following logic check that `transpose` is the only
                # function that falling back to pandas in the reduce operation flow.
                # Another warning comes from deprecated pandas.Int64Index usage.
                with pytest.warns(UserWarning) as warns:
                    res = getattr(df, method)()
                assert (
                    len(warns) == 2
                ), f"More than two warnings were arisen: len(warns) != 2 ({len(warns)} != 2)"
                message = warns[0].message.args[0]
                assert (
                    re.match(r".*transpose.*defaulting to pandas", message) is not None
                ), f"Expected DataFrame.transpose defaulting to pandas warning, got: {message}"
            else:
                res = getattr(df, method)()
            return res

        run_and_compare(applier, data=self.data, force_lazy=False)

    @pytest.mark.parametrize("data", [data, int_data])
    @pytest.mark.parametrize("dropna", bool_arg_values)
    def test_nunique(self, data, dropna):
        def applier(df, **kwargs):
            return df.nunique(dropna=dropna)

        run_and_compare(applier, data=data, force_lazy=False)


class TestMerge:
    data = {
        "a": [1, 2, 3, 6, 5, 4],
        "b": [10, 20, 30, 60, 50, 40],
        "e": [11, 22, 33, 66, 55, 44],
    }
    data2 = {
        "a": [4, 2, 3, 7, 1, 5],
        "b": [40, 20, 30, 70, 10, 50],
        "d": [4000, 2000, 3000, 7000, 1000, 5000],
    }
    on_values = ["a", ["a"], ["a", "b"], ["b", "a"], None]
    how_values = ["inner", "left"]

    @pytest.mark.parametrize("on", on_values)
    @pytest.mark.parametrize("how", how_values)
    @pytest.mark.parametrize("sort", [True, False])
    def test_merge(self, on, how, sort):
        def merge(lib, df1, df2, on, how, sort, **kwargs):
            return df1.merge(df2, on=on, how=how, sort=sort)

        run_and_compare(
            merge, data=self.data, data2=self.data2, on=on, how=how, sort=sort
        )

    def test_merge_non_str_column_name(self):
        def merge(lib, df1, df2, on, **kwargs):
            return df1.merge(df2, on=on, how="inner")

        run_and_compare(merge, data=[[1, 2], [3, 4]], data2=[[1, 2], [3, 4]], on=1)

    h2o_data = {
        "id1": ["id1", "id10", "id100", "id1000"],
        "id2": ["id2", "id20", "id200", "id2000"],
        "id3": ["id3", "id30", "id300", "id3000"],
        "id4": [4, 40, 400, 4000],
        "id5": [5, 50, 500, 5000],
        "id6": [6, 60, 600, 6000],
        "v1": [3.3, 4.4, 7.7, 8.8],
    }

    h2o_data_small = {
        "id1": ["id10", "id100", "id1000", "id10000"],
        "id4": [40, 400, 4000, 40000],
        "v2": [30.3, 40.4, 70.7, 80.8],
    }

    h2o_data_medium = {
        "id1": ["id10", "id100", "id1000", "id10000"],
        "id2": ["id20", "id200", "id2000", "id20000"],
        "id4": [40, 400, 4000, 40000],
        "id5": [50, 500, 5000, 50000],
        "v2": [30.3, 40.4, 70.7, 80.8],
    }

    h2o_data_big = {
        "id1": ["id10", "id100", "id1000", "id10000"],
        "id2": ["id20", "id200", "id2000", "id20000"],
        "id3": ["id30", "id300", "id3000", "id30000"],
        "id4": [40, 400, 4000, 40000],
        "id5": [50, 500, 5000, 50000],
        "id6": [60, 600, 6000, 60000],
        "v2": [30.3, 40.4, 70.7, 80.8],
    }

    def _get_h2o_df(self, data):
        df = pandas.DataFrame(data)
        if "id1" in data:
            df["id1"] = df["id1"].astype("category")
        if "id2" in data:
            df["id2"] = df["id2"].astype("category")
        if "id3" in data:
            df["id3"] = df["id3"].astype("category")
        return df

    # Currently OmniSci returns category as string columns
    # and therefore casted to category it would only have
    # values from actual data. In Pandas category would
    # have old values as well. Simply casting category
    # to string for somparison doesn't work because None
    # casted to category and back to strting becomes
    # "nan". So we cast everything to category and then
    # to string.
    def _fix_category_cols(self, df):
        if "id1" in df.columns:
            df["id1"] = df["id1"].astype("category")
            df["id1"] = df["id1"].astype(str)
        if "id1_x" in df.columns:
            df["id1_x"] = df["id1_x"].astype("category")
            df["id1_x"] = df["id1_x"].astype(str)
        if "id1_y" in df.columns:
            df["id1_y"] = df["id1_y"].astype("category")
            df["id1_y"] = df["id1_y"].astype(str)
        if "id2" in df.columns:
            df["id2"] = df["id2"].astype("category")
            df["id2"] = df["id2"].astype(str)
        if "id2_x" in df.columns:
            df["id2_x"] = df["id2_x"].astype("category")
            df["id2_x"] = df["id2_x"].astype(str)
        if "id2_y" in df.columns:
            df["id2_y"] = df["id2_y"].astype("category")
            df["id2_y"] = df["id2_y"].astype(str)
        if "id3" in df.columns:
            df["id3"] = df["id3"].astype("category")
            df["id3"] = df["id3"].astype(str)

    def test_h2o_q1(self):
        lhs = self._get_h2o_df(self.h2o_data)
        rhs = self._get_h2o_df(self.h2o_data_small)

        ref = lhs.merge(rhs, on="id1")
        self._fix_category_cols(ref)

        modin_lhs = pd.DataFrame(lhs)
        modin_rhs = pd.DataFrame(rhs)
        modin_res = modin_lhs.merge(modin_rhs, on="id1")

        exp = to_pandas(modin_res)
        self._fix_category_cols(exp)

        df_equals(ref, exp)

    def test_h2o_q2(self):
        lhs = self._get_h2o_df(self.h2o_data)
        rhs = self._get_h2o_df(self.h2o_data_medium)

        ref = lhs.merge(rhs, on="id2")
        self._fix_category_cols(ref)

        modin_lhs = pd.DataFrame(lhs)
        modin_rhs = pd.DataFrame(rhs)
        modin_res = modin_lhs.merge(modin_rhs, on="id2")

        exp = to_pandas(modin_res)
        self._fix_category_cols(exp)

        df_equals(ref, exp)

    def test_h2o_q3(self):
        lhs = self._get_h2o_df(self.h2o_data)
        rhs = self._get_h2o_df(self.h2o_data_medium)

        ref = lhs.merge(rhs, how="left", on="id2")
        self._fix_category_cols(ref)

        modin_lhs = pd.DataFrame(lhs)
        modin_rhs = pd.DataFrame(rhs)
        modin_res = modin_lhs.merge(modin_rhs, how="left", on="id2")

        exp = to_pandas(modin_res)
        self._fix_category_cols(exp)

        df_equals(ref, exp)

    def test_h2o_q4(self):
        lhs = self._get_h2o_df(self.h2o_data)
        rhs = self._get_h2o_df(self.h2o_data_medium)

        ref = lhs.merge(rhs, on="id5")
        self._fix_category_cols(ref)

        modin_lhs = pd.DataFrame(lhs)
        modin_rhs = pd.DataFrame(rhs)
        modin_res = modin_lhs.merge(modin_rhs, on="id5")

        exp = to_pandas(modin_res)
        self._fix_category_cols(exp)

        df_equals(ref, exp)

    def test_h2o_q5(self):
        lhs = self._get_h2o_df(self.h2o_data)
        rhs = self._get_h2o_df(self.h2o_data_big)

        ref = lhs.merge(rhs, on="id3")
        self._fix_category_cols(ref)

        modin_lhs = pd.DataFrame(lhs)
        modin_rhs = pd.DataFrame(rhs)
        modin_res = modin_lhs.merge(modin_rhs, on="id3")

        exp = to_pandas(modin_res)
        self._fix_category_cols(exp)

        df_equals(ref, exp)

    dt_data1 = {
        "id": [1, 2],
        "timestamp": pandas.to_datetime(["20000101", "20000201"], format="%Y%m%d"),
    }
    dt_data2 = {"id": [1, 2], "timestamp_year": [2000, 2000]}

    def test_merge_dt(self):
        def merge(df1, df2, **kwargs):
            df1["timestamp_year"] = df1["timestamp"].dt.year
            res = df1.merge(df2, how="left", on=["id", "timestamp_year"])
            res["timestamp_year"] = res["timestamp_year"].fillna(np.int64(-1))
            return res

        run_and_compare(merge, data=self.dt_data1, data2=self.dt_data2)

    left_data = {"a": [1, 2, 3, 4], "b": [10, 20, 30, 40], "c": [11, 12, 13, 14]}
    right_data = {"c": [1, 2, 3, 4], "b": [10, 20, 30, 40], "d": [100, 200, 300, 400]}

    @pytest.mark.parametrize("how", how_values)
    @pytest.mark.parametrize(
        "left_on, right_on", [["a", "c"], [["a", "b"], ["c", "b"]]]
    )
    def test_merge_left_right_on(self, how, left_on, right_on):
        def merge(df1, df2, how, left_on, right_on, **kwargs):
            return df1.merge(df2, how=how, left_on=left_on, right_on=right_on)

        run_and_compare(
            merge,
            data=self.left_data,
            data2=self.right_data,
            how=how,
            left_on=left_on,
            right_on=right_on,
        )
        run_and_compare(
            merge,
            data=self.right_data,
            data2=self.left_data,
            how=how,
            left_on=right_on,
            right_on=left_on,
        )


class TestBinaryOp:
    data = {
        "a": [1, 1, 1, 1, 1],
        "b": [10, 10, 10, 10, 10],
        "c": [100, 100, 100, 100, 100],
        "d": [1000, 1000, 1000, 1000, 1000],
    }
    data2 = {
        "a": [1, 1, 1, 1, 1],
        "f": [2, 2, 2, 2, 2],
        "b": [3, 3, 3, 3, 3],
        "d": [4, 4, 4, 4, 4],
    }
    fill_values = [None, 1]

    def test_binary_level(self):
        def applier(df1, df2, **kwargs):
            df2.index = generate_multiindex(len(df2))
            return df1.add(df2, level=1)

        # setting `force_lazy=False`, because we're expecting to fallback
        # to pandas in that case, which is not supported in lazy mode
        run_and_compare(applier, data=self.data, data2=self.data, force_lazy=False)

    def test_add_cst(self):
        def add(lib, df):
            return df + 1

        run_and_compare(add, data=self.data)

    def test_add_list(self):
        def add(lib, df):
            return df + [1, 2, 3, 4]

        run_and_compare(add, data=self.data)

    @pytest.mark.parametrize("fill_value", fill_values)
    def test_add_method_columns(self, fill_value):
        def add1(lib, df, fill_value):
            return df["a"].add(df["b"], fill_value=fill_value)

        def add2(lib, df, fill_value):
            return df[["a", "c"]].add(df[["b", "a"]], fill_value=fill_value)

        run_and_compare(add1, data=self.data, fill_value=fill_value)
        run_and_compare(add2, data=self.data, fill_value=fill_value)

    def test_add_columns(self):
        def add1(lib, df):
            return df["a"] + df["b"]

        def add2(lib, df):
            return df[["a", "c"]] + df[["b", "a"]]

        run_and_compare(add1, data=self.data)
        run_and_compare(add2, data=self.data)

    def test_add_columns_and_assign(self):
        def add(lib, df):
            df["sum"] = df["a"] + df["b"]
            return df

        run_and_compare(add, data=self.data)

    def test_add_columns_and_assign_to_existing(self):
        def add(lib, df):
            df["a"] = df["a"] + df["b"]
            return df

        run_and_compare(add, data=self.data)

    def test_mul_cst(self):
        def mul(lib, df):
            return df * 2

        run_and_compare(mul, data=self.data)

    def test_mul_list(self):
        def mul(lib, df):
            return df * [2, 3, 4, 5]

        run_and_compare(mul, data=self.data)

    @pytest.mark.parametrize("fill_value", fill_values)
    def test_mul_method_columns(self, fill_value):
        def mul1(lib, df, fill_value):
            return df["a"].mul(df["b"], fill_value=fill_value)

        def mul2(lib, df, fill_value):
            return df[["a", "c"]].mul(df[["b", "a"]], fill_value=fill_value)

        run_and_compare(mul1, data=self.data, fill_value=fill_value)
        run_and_compare(mul2, data=self.data, fill_value=fill_value)

    def test_mul_columns(self):
        def mul1(lib, df):
            return df["a"] * df["b"]

        def mul2(lib, df):
            return df[["a", "c"]] * df[["b", "a"]]

        run_and_compare(mul1, data=self.data)
        run_and_compare(mul2, data=self.data)

    def test_mod_cst(self):
        def mod(lib, df):
            return df % 2

        run_and_compare(mod, data=self.data)

    def test_mod_list(self):
        def mod(lib, df):
            return df % [2, 3, 4, 5]

        run_and_compare(mod, data=self.data)

    @pytest.mark.parametrize("fill_value", fill_values)
    def test_mod_method_columns(self, fill_value):
        def mod1(lib, df, fill_value):
            return df["a"].mod(df["b"], fill_value=fill_value)

        def mod2(lib, df, fill_value):
            return df[["a", "c"]].mod(df[["b", "a"]], fill_value=fill_value)

        run_and_compare(mod1, data=self.data, fill_value=fill_value)
        run_and_compare(mod2, data=self.data, fill_value=fill_value)

    def test_mod_columns(self):
        def mod1(lib, df):
            return df["a"] % df["b"]

        def mod2(lib, df):
            return df[["a", "c"]] % df[["b", "a"]]

        run_and_compare(mod1, data=self.data)
        run_and_compare(mod2, data=self.data)

    def test_truediv_cst(self):
        def truediv(lib, df):
            return df / 2

        run_and_compare(truediv, data=self.data)

    def test_truediv_list(self):
        def truediv(lib, df):
            return df / [1, 0.5, 0.2, 2.0]

        run_and_compare(truediv, data=self.data)

    @pytest.mark.parametrize("fill_value", fill_values)
    def test_truediv_method_columns(self, fill_value):
        def truediv1(lib, df, fill_value):
            return df["a"].truediv(df["b"], fill_value=fill_value)

        def truediv2(lib, df, fill_value):
            return df[["a", "c"]].truediv(df[["b", "a"]], fill_value=fill_value)

        run_and_compare(truediv1, data=self.data, fill_value=fill_value)
        run_and_compare(truediv2, data=self.data, fill_value=fill_value)

    def test_truediv_columns(self):
        def truediv1(lib, df):
            return df["a"] / df["b"]

        def truediv2(lib, df):
            return df[["a", "c"]] / df[["b", "a"]]

        run_and_compare(truediv1, data=self.data)
        run_and_compare(truediv2, data=self.data)

    def test_floordiv_cst(self):
        def floordiv(lib, df):
            return df // 2

        run_and_compare(floordiv, data=self.data)

    def test_floordiv_list(self):
        def floordiv(lib, df):
            return df // [1, 0.54, 0.24, 2.01]

        run_and_compare(floordiv, data=self.data)

    @pytest.mark.parametrize("fill_value", fill_values)
    def test_floordiv_method_columns(self, fill_value):
        def floordiv1(lib, df, fill_value):
            return df["a"].floordiv(df["b"], fill_value=fill_value)

        def floordiv2(lib, df, fill_value):
            return df[["a", "c"]].floordiv(df[["b", "a"]], fill_value=fill_value)

        run_and_compare(floordiv1, data=self.data, fill_value=fill_value)
        run_and_compare(floordiv2, data=self.data, fill_value=fill_value)

    def test_floordiv_columns(self):
        def floordiv1(lib, df):
            return df["a"] // df["b"]

        def floordiv2(lib, df):
            return df[["a", "c"]] // df[["b", "a"]]

        run_and_compare(floordiv1, data=self.data)
        run_and_compare(floordiv2, data=self.data)

    cmp_data = {
        "a": [1, 2, 3, 4, 5],
        "b": [10, 20, 30, 40, 50],
        "c": [50.0, 40.0, 30.1, 20.0, 10.0],
    }
    cmp_fn_values = ["eq", "ne", "le", "lt", "ge", "gt"]

    @pytest.mark.parametrize("cmp_fn", cmp_fn_values)
    def test_cmp_cst(self, cmp_fn):
        def cmp1(df, cmp_fn, **kwargs):
            return getattr(df["a"], cmp_fn)(3)

        def cmp2(df, cmp_fn, **kwargs):
            return getattr(df, cmp_fn)(30)

        run_and_compare(cmp1, data=self.cmp_data, cmp_fn=cmp_fn)
        run_and_compare(cmp2, data=self.cmp_data, cmp_fn=cmp_fn)

    @pytest.mark.parametrize("cmp_fn", cmp_fn_values)
    def test_cmp_list(self, cmp_fn):
        def cmp(df, cmp_fn, **kwargs):
            return getattr(df, cmp_fn)([3, 30, 30.1])

        run_and_compare(cmp, data=self.cmp_data, cmp_fn=cmp_fn)

    @pytest.mark.parametrize("cmp_fn", cmp_fn_values)
    def test_cmp_cols(self, cmp_fn):
        def cmp1(df, cmp_fn, **kwargs):
            return getattr(df["b"], cmp_fn)(df["c"])

        def cmp2(df, cmp_fn, **kwargs):
            return getattr(df[["b", "c"]], cmp_fn)(df[["a", "b"]])

        run_and_compare(cmp1, data=self.cmp_data, cmp_fn=cmp_fn)
        run_and_compare(cmp2, data=self.cmp_data, cmp_fn=cmp_fn)

    @pytest.mark.parametrize("cmp_fn", cmp_fn_values)
    @pytest.mark.parametrize("value", [2, 2.2, "a"])
    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    def test_cmp_mixed_types(self, cmp_fn, value, data):
        def cmp(df, cmp_fn, value, **kwargs):
            return getattr(df, cmp_fn)(value)

        run_and_compare(cmp, data=data, cmp_fn=cmp_fn, value=value)

    def test_filter_dtypes(self):
        def filter(df, **kwargs):
            return df[df.a < 4].dtypes

        run_and_compare(filter, data=self.cmp_data)

    @pytest.mark.xfail(
        reason="Requires fix in OmniSci: https://github.com/intel-ai/omniscidb/pull/178"
    )
    def test_filter_empty_result(self):
        def filter(df, **kwargs):
            return df[df.a < 0]

        run_and_compare(filter, data=self.cmp_data)

    def test_complex_filter(self):
        def filter_and(df, **kwargs):
            return df[(df.a < 5) & (df.b > 20)]

        def filter_or(df, **kwargs):
            return df[(df.a < 3) | (df.b > 40)]

        run_and_compare(filter_and, data=self.cmp_data)
        run_and_compare(filter_or, data=self.cmp_data)


class TestDateTime:
    datetime_data = {
        "a": [1, 1, 2, 2],
        "b": [11, 21, 12, 11],
        "c": pandas.to_datetime(
            ["20190902", "20180913", "20190921", "20180903"], format="%Y%m%d"
        ),
        "d": pandas.to_datetime(
            [
                "2018-10-26 12:00",
                "2018-10-26 13:00:15",
                "2020-10-26 04:00:15",
                "2020-10-26",
            ]
        ),
    }

    def test_dt_year(self):
        def dt_year(df, **kwargs):
            return df["c"].dt.year

        run_and_compare(dt_year, data=self.datetime_data)

    def test_dt_month(self):
        def dt_month(df, **kwargs):
            return df["c"].dt.month

        run_and_compare(dt_month, data=self.datetime_data)

    def test_dt_day(self):
        def dt_day(df, **kwargs):
            return df["c"].dt.day

        run_and_compare(dt_day, data=self.datetime_data)

    def test_dt_hour(self):
        def dt_hour(df, **kwargs):
            return df["d"].dt.hour

        run_and_compare(dt_hour, data=self.datetime_data)


class TestCategory:
    data = {
        "a": ["str1", "str2", "str1", "str3", "str2", None],
    }

    def test_cat_codes(self):
        pandas_df = pandas.DataFrame(self.data)
        pandas_df["a"] = pandas_df["a"].astype("category")

        modin_df = pd.DataFrame(pandas_df)

        modin_df["a"] = modin_df["a"].cat.codes
        exp = to_pandas(modin_df)

        pandas_df["a"] = pandas_df["a"].cat.codes

        df_equals(pandas_df, exp)


class TestSort:
    data = {
        "a": [1, 2, 5, 2, 5, 4, 4, 5, 2],
        "b": [1, 2, 3, 6, 5, 1, 4, 5, 3],
        "c": [5, 4, 2, 3, 1, 1, 4, 5, 6],
        "d": ["1", "4", "3", "2", "1", "6", "7", "5", "0"],
    }
    data_nulls = {
        "a": [1, 2, 5, 2, 5, 4, 4, None, 2],
        "b": [1, 2, 3, 6, 5, None, 4, 5, 3],
        "c": [None, 4, 2, 3, 1, 1, 4, 5, 6],
    }
    data_multiple_nulls = {
        "a": [1, 2, None, 2, 5, 4, 4, None, 2],
        "b": [1, 2, 3, 6, 5, None, 4, 5, None],
        "c": [None, 4, 2, None, 1, 1, 4, 5, 6],
    }
    cols_values = ["a", ["a", "b"], ["b", "a"], ["c", "a", "b"]]
    index_cols_values = [None, "a", ["a", "b"]]
    ascending_values = [True, False]
    ascending_list_values = [[True, False], [False, True]]
    na_position_values = ["first", "last"]

    @pytest.mark.parametrize("cols", cols_values)
    @pytest.mark.parametrize("ignore_index", bool_arg_values)
    @pytest.mark.parametrize("ascending", ascending_values)
    @pytest.mark.parametrize("index_cols", index_cols_values)
    def test_sort_cols(self, cols, ignore_index, index_cols, ascending):
        def sort(df, cols, ignore_index, index_cols, ascending, **kwargs):
            if index_cols:
                df = df.set_index(index_cols)
            return df.sort_values(cols, ignore_index=ignore_index, ascending=ascending)

        run_and_compare(
            sort,
            data=self.data,
            cols=cols,
            ignore_index=ignore_index,
            index_cols=index_cols,
            ascending=ascending,
            # we're expecting to fallback to pandas in that case,
            # which is not supported in lazy mode
            force_lazy=(index_cols is None),
        )

    @pytest.mark.parametrize("ascending", ascending_list_values)
    def test_sort_cols_asc_list(self, ascending):
        def sort(df, ascending, **kwargs):
            return df.sort_values(["a", "b"], ascending=ascending)

        run_and_compare(
            sort,
            data=self.data,
            ascending=ascending,
        )

    @pytest.mark.parametrize("ascending", ascending_values)
    def test_sort_cols_str(self, ascending):
        def sort(df, ascending, **kwargs):
            return df.sort_values("d", ascending=ascending)

        run_and_compare(
            sort,
            data=self.data,
            ascending=ascending,
        )

    @pytest.mark.parametrize("cols", cols_values)
    @pytest.mark.parametrize("ascending", ascending_values)
    @pytest.mark.parametrize("na_position", na_position_values)
    def test_sort_cols_nulls(self, cols, ascending, na_position):
        def sort(df, cols, ascending, na_position, **kwargs):
            return df.sort_values(cols, ascending=ascending, na_position=na_position)

        run_and_compare(
            sort,
            data=self.data_nulls,
            cols=cols,
            ascending=ascending,
            na_position=na_position,
        )

    # Issue #1767 - rows order is not preserved for NULL keys
    # @pytest.mark.parametrize("cols", cols_values)
    # @pytest.mark.parametrize("ascending", ascending_values)
    # @pytest.mark.parametrize("na_position", na_position_values)
    # def test_sort_cols_multiple_nulls(self, cols, ascending, na_position):
    #    def sort(df, cols, ascending, na_position, **kwargs):
    #        return df.sort_values(cols, ascending=ascending, na_position=na_position)
    #
    #    run_and_compare(
    #        sort,
    #        data=self.data_multiple_nulls,
    #        cols=cols,
    #        ascending=ascending,
    #        na_position=na_position,
    #    )


class TestBadData:
    bad_for_arrow = {
        "a": ["a", [[1, 2], [3]], [3, 4]],
        "b": ["b", [1, 2], [3, 4]],
        "c": ["1", "2", 3],
    }
    bad_for_omnisci = {
        "b": [[1, 2], [3, 4], [5, 6]],
        "c": ["1", "2", "3"],
    }
    ok_data = {"d": np.arange(3), "e": np.arange(3), "f": np.arange(3)}

    def _get_pyarrow_table(self, obj):
        if not isinstance(obj, (pandas.DataFrame, pandas.Series)):
            obj = pandas.DataFrame(obj)

        return pyarrow.Table.from_pandas(obj)

    @pytest.mark.parametrize("data", [bad_for_arrow, bad_for_omnisci])
    def test_construct(self, data):
        def applier(df, *args, **kwargs):
            return repr(df)

        run_and_compare(applier, data=data, force_lazy=False)

    def test_from_arrow(self):
        at = self._get_pyarrow_table(self.bad_for_omnisci)
        pd_df = pandas.DataFrame(self.bad_for_omnisci)
        md_df = pd.utils.from_arrow(at)

        # force materialization
        repr(md_df)
        df_equals(md_df, pd_df)

    @pytest.mark.parametrize("data", [bad_for_arrow, bad_for_omnisci])
    def test_methods(self, data):
        def applier(df, *args, **kwargs):
            return df.T.drop(columns=[0])

        run_and_compare(applier, data=data, force_lazy=False)

    def test_with_normal_frame(self):
        def applier(df1, df2, *args, **kwargs):
            return df2.join(df1)

        run_and_compare(
            applier, data=self.bad_for_omnisci, data2=self.ok_data, force_lazy=False
        )

    def test_heterogenous_fillna(self):
        def fillna(df, **kwargs):
            return df["d"].fillna("a")

        run_and_compare(fillna, data=self.ok_data, force_lazy=False)

    @pytest.mark.parametrize(
        "md_df_constructor",
        [
            pytest.param(pd.DataFrame, id="from_pandas_dataframe"),
            pytest.param(
                lambda pd_df: from_arrow(pyarrow.Table.from_pandas(pd_df)),
                id="from_pyarrow_table",
            ),
        ],
    )
    def test_uint(self, md_df_constructor):
        """
        Verify that unsigned integer data could be imported-exported via OmniSci with no errors.

        Originally, OmniSci does not support unsigned integers, there's a logic in Modin that
        upcasts unsigned types to the compatible ones prior importing to OmniSci.
        """
        pd_df = pandas.DataFrame(
            {
                "uint8_in_int_bounds": np.array([1, 2, 3], dtype="uint8"),
                "uint8_out-of_int_bounds": np.array(
                    [(2**8) - 1, (2**8) - 2, (2**8) - 3], dtype="uint8"
                ),
                "uint16_in_int_bounds": np.array([1, 2, 3], dtype="uint16"),
                "uint16_out-of_int_bounds": np.array(
                    [(2**16) - 1, (2**16) - 2, (2**16) - 3], dtype="uint16"
                ),
                "uint32_in_int_bounds": np.array([1, 2, 3], dtype="uint32"),
                "uint32_out-of_int_bounds": np.array(
                    [(2**32) - 1, (2**32) - 2, (2**32) - 3], dtype="uint32"
                ),
                "uint64_in_int_bounds": np.array([1, 2, 3], dtype="uint64"),
            }
        )
        md_df = md_df_constructor(pd_df)

        with ForceOmnisciImport(md_df) as instance:
            md_df_exported = instance.export_frames()[0]
            result = md_df_exported.values
            reference = pd_df.values
            np.testing.assert_array_equal(result, reference)

    @pytest.mark.parametrize(
        "md_df_constructor",
        [
            pytest.param(pd.DataFrame, id="from_pandas_dataframe"),
            pytest.param(
                lambda pd_df: from_arrow(pyarrow.Table.from_pandas(pd_df)),
                id="from_pyarrow_table",
            ),
        ],
    )
    def test_uint_overflow(self, md_df_constructor):
        """
        Verify that the exception is arisen when overflow occurs due to 'uint -> int' compatibility conversion.

        Originally, OmniSci does not support unsigned integers, there's a logic in Modin that upcasts
        unsigned types to the compatible ones prior importing to OmniSci. This test ensures that the
        error is arisen when such conversion causes a data loss.
        """
        md_df = md_df_constructor(
            pandas.DataFrame(
                {
                    "col": np.array(
                        [(2**64) - 1, (2**64) - 2, (2**64) - 3], dtype="uint64"
                    )
                }
            )
        )

        with pytest.raises(OverflowError):
            with ForceOmnisciImport(md_df):
                pass


class TestDropna:
    data = {
        "col1": [1, 2, None, 2, 1],
        "col2": [None, 3, None, 2, 1],
        "col3": [2, 3, 4, None, 5],
        "col4": [1, 2, 3, 4, 5],
    }

    @pytest.mark.parametrize("subset", [None, ["col1", "col2"]])
    @pytest.mark.parametrize("how", ["all", "any"])
    def test_dropna(self, subset, how):
        def applier(df, *args, **kwargs):
            return df.dropna(subset=subset, how=how)

        run_and_compare(applier, data=self.data)

    def test_dropna_multiindex(self):
        index = generate_multiindex(len(self.data["col1"]))

        md_df = pd.DataFrame(self.data, index=index)
        pd_df = pandas.DataFrame(self.data, index=index)

        md_res = md_df.dropna()._to_pandas()
        pd_res = pd_df.dropna()

        # HACK: all strings in OmniSci considered to be categories, that breaks
        # checks for equality with pandas, this line discards category dtype
        md_res.index = pandas.MultiIndex.from_tuples(
            md_res.index.values, names=md_res.index.names
        )

        df_equals(md_res, pd_res)

    @pytest.mark.skip("Dropna logic for GroupBy is disabled for now")
    @pytest.mark.parametrize("by", ["col1", ["col1", "col2"], ["col1", "col4"]])
    @pytest.mark.parametrize("dropna", [True, False])
    def test_dropna_groupby(self, by, dropna):
        def applier(df, *args, **kwargs):
            # OmniSci engine preserves NaNs at the result of groupby,
            # so replacing NaNs with '0' to match with Pandas.
            # https://github.com/modin-project/modin/issues/2878
            return df.groupby(by=by, dropna=dropna).sum().fillna(0)

        run_and_compare(applier, data=self.data)


class TestUnsupportedColumns:
    @pytest.mark.parametrize(
        "data,is_good",
        [
            [["1", "2", None, "2", "1"], True],
            [[None, "3", None, "2", "1"], True],
            [[1, "2", None, "2", "1"], False],
            [[None, 3, None, "2", "1"], False],
        ],
    )
    def test_unsupported_columns(self, data, is_good):
        pandas_df = pandas.DataFrame({"col": data})
        obj, bad_cols = OmnisciOnNativeDataframePartitionManager._get_unsupported_cols(
            pandas_df
        )
        if is_good:
            assert obj and not bad_cols
        else:
            assert not obj and bad_cols == ["col"]


class TestConstructor:
    @pytest.mark.parametrize(
        "index",
        [
            None,
            pandas.Index([1, 2, 3]),
            pandas.MultiIndex.from_tuples([(1, 1), (2, 2), (3, 3)]),
        ],
    )
    def test_shape_hint_detection(self, index):
        df = pd.DataFrame({"a": [1, 2, 3]}, index=index)
        assert df._query_compiler._shape_hint == "column"

        transposed_data = df._to_pandas().T.to_dict()
        df = pd.DataFrame(transposed_data)
        assert df._query_compiler._shape_hint == "row"

        df = pd.DataFrame({"a": [1, 2, 3], "b": [1, 2, 3]}, index=index)
        assert df._query_compiler._shape_hint is None

        df = pd.DataFrame({"a": [1]}, index=None if index is None else index[:1])
        assert df._query_compiler._shape_hint == "column"

    def test_shape_hint_detection_from_arrow(self):
        at = pyarrow.Table.from_pydict({"a": [1, 2, 3]})
        df = pd.utils.from_arrow(at)
        assert df._query_compiler._shape_hint == "column"

        at = pyarrow.Table.from_pydict({"a": [1], "b": [2], "c": [3]})
        df = pd.utils.from_arrow(at)
        assert df._query_compiler._shape_hint == "row"

        at = pyarrow.Table.from_pydict({"a": [1, 2, 3], "b": [1, 2, 3]})
        df = pd.utils.from_arrow(at)
        assert df._query_compiler._shape_hint is None

        at = pyarrow.Table.from_pydict({"a": [1]})
        df = pd.utils.from_arrow(at)
        assert df._query_compiler._shape_hint == "column"


class TestArrowExecution:
    data1 = {"a": [1, 2, 3], "b": [3, 4, 5], "c": [6, 7, 8]}
    data2 = {"a": [1, 2, 3], "d": [3, 4, 5], "e": [6, 7, 8]}
    data3 = {"a": [4, 5, 6], "b": [6, 7, 8], "c": [9, 10, 11]}

    def test_drop_rename_concat(self):
        def drop_rename_concat(df1, df2, lib, **kwargs):
            df1 = df1.rename(columns={"a": "new_a", "c": "new_b"})
            df1 = df1.drop(columns="b")
            df2 = df2.rename(columns={"a": "new_a", "d": "new_b"})
            df2 = df2.drop(columns="e")
            return lib.concat([df1, df2], ignore_index=True)

        run_and_compare(
            drop_rename_concat,
            data=self.data1,
            data2=self.data2,
            force_arrow_execute=True,
        )

    def test_empty_transform(self):
        def apply(df, **kwargs):
            return df + 1

        run_and_compare(apply, data={}, force_arrow_execute=True)

    def test_append(self):
        def apply(df1, df2, **kwargs):
            tmp = df1.append(df2)
            return tmp

        run_and_compare(
            apply, data=self.data1, data2=self.data3, force_arrow_execute=True
        )


if __name__ == "__main__":
    pytest.main(["-v", __file__])
