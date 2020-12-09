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

from modin.config import IsExperimental, Engine, Backend

IsExperimental.put(True)
Engine.put("ray")
Backend.put("omnisci")

import modin.pandas as pd
from modin.pandas.test.utils import (
    df_equals,
    bool_arg_values,
    to_pandas,
    test_data_values,
    test_data_keys,
    generate_multiindex,
    eval_general,
)


def set_execution_mode(frame, mode, recursive=False):
    if isinstance(frame, (pd.Series, pd.DataFrame)):
        frame = frame._query_compiler._modin_frame
    frame._force_execution_mode = mode
    if recursive and hasattr(frame._op, "input"):
        for child in frame._op.input:
            set_execution_mode(child, mode, True)


def run_and_compare(
    fn,
    data,
    data2=None,
    force_lazy=True,
    force_arrow_execute=False,
    allow_subqueries=False,
    **kwargs
):
    def run_modin(
        fn,
        data,
        data2,
        force_lazy,
        force_arrow_execute,
        allow_subqueries,
        constructor_kwargs,
        **kwargs
    ):
        kwargs["df1"] = pd.DataFrame(data, **constructor_kwargs)
        kwargs["df2"] = pd.DataFrame(data2, **constructor_kwargs)
        kwargs["df"] = kwargs["df1"]

        if force_lazy:
            set_execution_mode(kwargs["df1"], "lazy")
            set_execution_mode(kwargs["df2"], "lazy")
        elif force_arrow_execute:
            set_execution_mode(kwargs["df1"], "arrow")
            set_execution_mode(kwargs["df2"], "arrow")

        exp_res = fn(lib=pd, **kwargs)

        if force_arrow_execute:
            set_execution_mode(exp_res, "arrow", allow_subqueries)
        elif force_lazy:
            set_execution_mode(exp_res, None, allow_subqueries)

        return exp_res

    constructor_kwargs = kwargs.pop("constructor_kwargs", {})
    try:
        kwargs["df1"] = pandas.DataFrame(data, **constructor_kwargs)
        kwargs["df2"] = pandas.DataFrame(data2, **constructor_kwargs)
        kwargs["df"] = kwargs["df1"]
        ref_res = fn(lib=pandas, **kwargs)
    except Exception as e:
        with pytest.raises(type(e)):
            exp_res = run_modin(
                fn=fn,
                data=data,
                data2=data2,
                force_lazy=force_lazy,
                force_arrow_execute=force_arrow_execute,
                allow_subqueries=allow_subqueries,
                constructor_kwargs=constructor_kwargs,
                **kwargs
            )
            _ = exp_res.index
    else:
        exp_res = run_modin(
            fn=fn,
            data=data,
            data2=data2,
            force_lazy=force_lazy,
            force_arrow_execute=force_arrow_execute,
            allow_subqueries=allow_subqueries,
            constructor_kwargs=constructor_kwargs,
            **kwargs
        )
        df_equals(ref_res, exp_res)


class TestCSV:
    root = os.path.abspath(__file__ + "/.." * 6)  # root of modin repo

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
        """ check with the following arguments: names, dtype, skiprows, delimiter """
        csv_file = os.path.join(self.root, "modin/pandas/test/data", "test_usecols.csv")

        for kwargs in (
            {"delimiter": ","},
            {"sep": None},
            {"skiprows": 1, "names": ["A", "B", "C", "D", "E"]},
            {"dtype": {"a": "int32", "e": "string"}},
            {"dtype": {"a": np.dtype("int32"), "b": np.dtype("int64"), "e": "string"}},
        ):
            rp = pandas.read_csv(csv_file, **kwargs)
            rm = to_pandas(pd.read_csv(csv_file, engine="arrow", **kwargs))
            df_equals(rp, rm)

    def test_housing_csv(self):
        csv_file = os.path.join(self.root, "examples/data/boston_housing.csv")
        for kwargs in (
            {
                "skiprows": 1,
                "names": self.boston_housing_names,
                "dtype": self.boston_housing_dtypes,
            },
        ):
            rp = pandas.read_csv(csv_file, **kwargs)
            rm = to_pandas(pd.read_csv(csv_file, engine="arrow", **kwargs))
            assert rp is not None
            assert rm is not None
            # TODO: df_equals(rp, rm)  #  needs inexact comparison

    def test_time_parsing(self):
        csv_file = os.path.join(
            self.root, "modin/pandas/test/data", "test_time_parsing.csv"
        )
        for kwargs in (
            {
                "skiprows": 1,
                "names": [
                    "timestamp",
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
            rm = to_pandas(pd.read_csv(csv_file, engine="arrow", **kwargs))
            df_equals(rm["timestamp"].dt.year, rp["timestamp"].dt.year)

    def test_csv_fillna(self):
        csv_file = os.path.join(self.root, "examples/data/boston_housing.csv")
        for kwargs in (
            {
                "skiprows": 1,
                "names": self.boston_housing_names,
                "dtype": self.boston_housing_dtypes,
            },
        ):
            rp = pandas.read_csv(csv_file, **kwargs)
            rp = rp["CRIM"].fillna(1000)
            rm = pd.read_csv(csv_file, engine="arrow", **kwargs)
            rm = rm["CRIM"].fillna(1000)
            df_equals(rp, rm)

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

        df_equals(ref, exp)

    @pytest.mark.parametrize("names", [None, ["a", "b", "c", "d", "e"]])
    @pytest.mark.parametrize("header", [None, 0])
    def test_from_csv(self, header, names):
        csv_file = os.path.join(self.root, "modin/pandas/test/data", "test_usecols.csv")
        kwargs = {
            "header": header,
            "names": names,
        }

        pandas_df = pandas.read_csv(csv_file, **kwargs)
        modin_df = pd.read_csv(csv_file, **kwargs)

        df_equals(modin_df, pandas_df)

    @pytest.mark.parametrize("kwargs", [{"sep": "|"}, {"delimiter": "|"}])
    def test_sep_delimiter(self, kwargs):
        csv_file = os.path.join(self.root, "modin/pandas/test/data", "test_delim.csv")

        pandas_df = pandas.read_csv(csv_file, **kwargs)
        modin_df = pd.read_csv(csv_file, **kwargs)

        df_equals(modin_df, pandas_df)

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

        df_equals(modin_df, pandas_df)


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

        run_and_compare(concat, data=self.data, data2=self.data2)

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


class TestGroupby:
    data = {
        "a": [1, 1, 2, 2, 2],
        "b": [11, 21, 12, 22, 32],
        "c": [101, 201, 102, 202, 302],
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

    @pytest.mark.xfail(
        reason="Currently mean() passes a lambda into backend which cannot be executed on omnisci backend"
    )
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

    def test_groupby_agg_count(self):
        def groupby(df, **kwargs):
            return df.groupby("a").agg({"b": "count"})

        run_and_compare(groupby, data=self.data)

    def test_groupby_agg_size(self):
        def groupby(df, **kwargs):
            return df.groupby("a").agg({"b": "size"})

        run_and_compare(groupby, data=self.data)

    @pytest.mark.xfail(
        reason="Function specified as a string should be passed into backend API, but currently it is transformed into a lambda"
    )
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
            # TODO: For now we can't do such groupby by first column since modin use that
            #      column as aggregation one by default. We don't support such cases at
            #      at the moment, this will be handled later
            # ref = df.groupby("a").size()
            return df.groupby("b").size()

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

    @pytest.mark.parametrize("cols", ["a", "d"])
    @pytest.mark.parametrize("dropna", [True, False])
    @pytest.mark.parametrize("sort", [True])
    @pytest.mark.parametrize("ascending", [True, False])
    def test_value_counts(self, cols, dropna, sort, ascending):
        def value_counts(df, cols, dropna, sort, ascending, **kwargs):
            return df[cols].value_counts(dropna=dropna, sort=sort, ascending=ascending)

        run_and_compare(
            value_counts,
            data=self.data,
            cols=cols,
            dropna=dropna,
            sort=sort,
            ascending=ascending,
        )


class TestMerge:
    data = {
        "a": [1, 2, 3],
        "b": [10, 20, 30],
        "e": [11, 22, 33],
    }
    data2 = {
        "a": [4, 2, 3],
        "b": [40, 20, 30],
        "d": [4000, 2000, 3000],
    }
    on_values = ["a", ["a"], ["a", "b"], ["b", "a"]]
    how_values = ["inner", "left"]

    @pytest.mark.parametrize("on", on_values)
    @pytest.mark.parametrize("how", how_values)
    def test_merge(self, on, how):
        def merge(lib, df1, df2, on, how):
            return df1.merge(df2, on=on, how=how)

        run_and_compare(merge, data=self.data, data2=self.data2, on=on, how=how)

    @pytest.mark.parametrize("how", how_values)
    def test_default_merge(self, how):
        def default_merge(lib, df1, df2, how):
            return df1.merge(df2, how=how)

        run_and_compare(default_merge, data=self.data, data2=self.data2, how=how)

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


class TestDateTime:
    datetime_data = {
        "a": [1, 1, 2, 2],
        "b": [11, 21, 12, 11],
        "c": pandas.to_datetime(
            ["20190902", "20180913", "20190921", "20180903"], format="%Y%m%d"
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


if __name__ == "__main__":
    pytest.main(["-v", __file__])
