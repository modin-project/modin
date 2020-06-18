import os

os.environ["MODIN_EXPERIMENTAL"] = "True"
os.environ["MODIN_ENGINE"] = "ray"
os.environ["MODIN_BACKEND"] = "omnisci"

if os.environ["OMNISCI_SERVER"] is None:
    raise RuntimeError("OMNISCI_SERVER variable must be initialized")

import modin.pandas as mpd
import pandas as pd
import numpy as np

import pytest
from modin.pandas.test.utils import (
    df_equals,
    bool_arg_values,
    join_type_keys,
    to_pandas,
)


def run_and_compare(fn, data, data2=None, *args, **kwargs):
    if data2 is None:
        pandas_df = pd.DataFrame(data, columns=list(data.keys()))
        ref_res = fn(df=pandas_df, lib=pd, *args, **kwargs)
        modin_df = mpd.DataFrame(data, columns=list(data.keys()))
        exp_res = fn(df=modin_df, lib=mpd, *args, **kwargs)
    else:
        pandas_df1 = pd.DataFrame(data, columns=list(data.keys()))
        pandas_df2 = pd.DataFrame(data2, columns=list(data2.keys()))
        ref_res = fn(df1=pandas_df1, df2=pandas_df2, lib=pd, *args, **kwargs)
        modin_df1 = mpd.DataFrame(data, columns=list(data.keys()))
        modin_df2 = mpd.DataFrame(data2, columns=list(data2.keys()))
        exp_res = fn(df1=modin_df1, df2=modin_df2, lib=mpd, *args, **kwargs)
    df_equals(ref_res, exp_res)


class TestMasks:
    data = {"a": [1, 1, None], "b": [None, None, 2], "c": [3, None, None]}
    cols_values = ["a", ["a", "b"], ["a", "b", "c"]]

    @pytest.mark.parametrize("cols", cols_values)
    def test_projection(self, cols):
        def projection(df, cols, **kwargs):
            return df[cols]

        run_and_compare(projection, data=self.data, cols=cols)

    def test_drop(self):
        pandas_df = pd.DataFrame(self.data)
        modin_df = pd.DataFrame(self.data)

        pandas_df = pandas_df.drop(columns="a")
        modin_df = modin_df.drop(columns="a")
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
        pandas_df = pd.DataFrame(self.data)
        modin_df = mpd.DataFrame(self.data)
        pandas_df["f"] = pandas_df["a"]
        modin_df["f"] = modin_df["a"]
        df_equals(pandas_df, modin_df)

    def test_insert(self):
        def insert(df, **kwargs):
            df["new_int8"] = np.int8(10)
            df["new_int16"] = np.int16(10)
            df["new_int32"] = np.int32(10)
            df["new_int64"] = np.int64(10)
            df["new_int"] = 10
            df["new_float"] = 5.5
            return df

        run_and_compare(insert, data=self.data)


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
    def test_groupby_proj_sum(self, cols, as_index):
        def groupby_sum(df, cols, as_index, **kwargs):
            return df.groupby(cols, as_index=as_index).c.sum()

        run_and_compare(groupby_sum, data=self.data, cols=cols, as_index=as_index)

    def test_groupby_agg_count(self):
        df = pd.DataFrame(self.data)
        ref = df.groupby("a").agg({"b": "count"})

        modin_df = mpd.DataFrame(self.data)
        modin_df = modin_df.groupby("a").agg({"b": "count"})

        exp = to_pandas(modin_df)

        df_equals(ref, exp)

    def test_groupby_agg_size(self):
        df = pd.DataFrame(self.data)
        ref = df.groupby("a").agg({"b": "size"})

        modin_df = mpd.DataFrame(self.data)
        modin_df = modin_df.groupby("a").agg({"b": "size"})

        exp = to_pandas(modin_df)

        df_equals(ref, exp)        

    taxi_data = {
        "a": [1, 1, 2, 2],
        "b": [11, 21, 12, 11],
        "c": pd.to_datetime(
            ["20190902", "20180913", "20190921", "20180903"], format="%Y%m%d"
        ),
        "d": [11.5, 21.2, 12.8, 13.4],
    }

    # TODO: emulate taxi queries with group by category types when we have loading
    #       using arrow
    #       Another way of doing taxi q1 is
    #       res = df.groupby("cab_type").size() - this should be tested later as well
    def test_taxi_q1(self):
        df = pd.DataFrame(self.taxi_data)
        ref = df.groupby("a").size()

        modin_df = mpd.DataFrame(self.taxi_data)
        modin_df = modin_df.groupby("a").size()

        exp = to_pandas(modin_df)

        df_equals(ref, exp)

    def test_taxi_q2(self):
        df = pd.DataFrame(self.taxi_data)
        ref = df.groupby("a").agg({"b": "mean"})

        modin_df = mpd.DataFrame(self.taxi_data)
        modin_df = modin_df.groupby("a").agg({"b": "mean"})

        exp = to_pandas(modin_df)

        df_equals(ref, exp)

    def test_dt_year(self):
        df = pd.DataFrame(self.taxi_data)
        ref = df["c"].dt.year

        modin_df = mpd.DataFrame(self.taxi_data)
        modin_df = modin_df["c"].dt.year

        exp = to_pandas(modin_df)

        df_equals(ref, exp)

    @pytest.mark.parametrize("as_index", bool_arg_values)
    def test_taxi_q3(self, as_index):
        df = pd.DataFrame(self.taxi_data)
        ref = df.groupby(["b", df["c"].dt.year], as_index=as_index).size()

        modin_df = mpd.DataFrame(self.taxi_data)
        modin_df = modin_df.groupby(
            ["b", modin_df["c"].dt.year], as_index=as_index).size()

        exp = to_pandas(modin_df)

        df_equals(ref, exp)            

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
        df = pd.DataFrame(self.taxi_data)
        ref = df["d"].astype("int")

        modin_df = mpd.DataFrame(self.taxi_data)
        modin_df = modin_df["d"].astype("int")

        exp = to_pandas(modin_df)

        df_equals(ref, exp)

    def test_df_astype(self):
        df = pd.DataFrame(self.taxi_data)
        ref = df.astype({"b": "float", "d": "int"})

        modin_df = mpd.DataFrame(self.taxi_data)
        modin_df = modin_df.astype({"b": "float", "d": "int"})

        exp = to_pandas(modin_df)

        df_equals(ref, exp)

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
        df = pd.DataFrame(self.h2o_data)
        df["id1"] = df["id1"].astype("category")
        df["id2"] = df["id2"].astype("category")
        df["id3"] = df["id3"].astype("category")
        return df

    def test_h2o_q1(self):
        df = self._get_h2o_df()

        ref = df.groupby(["id1"], observed=True).agg({"v1": "sum"})
        ref.reset_index(inplace=True)

        modin_df = mpd.DataFrame(df)
        modin_df = modin_df.groupby(["id1"], observed=True, as_index=False).agg(
            {"v1": "sum"}
        )

        exp = to_pandas(modin_df)
        exp["id1"] = exp["id1"].astype("category")

        df_equals(ref, exp)

    def test_h2o_q2(self):
        df = self._get_h2o_df()

        ref = df.groupby(["id1", "id2"], observed=True).agg({"v1": "sum"})
        ref.reset_index(inplace=True)

        modin_df = mpd.DataFrame(df)
        modin_df = modin_df.groupby(["id1", "id2"], observed=True, as_index=False).agg(
            {"v1": "sum"}
        )

        exp = to_pandas(modin_df)
        exp["id1"] = exp["id1"].astype("category")
        exp["id2"] = exp["id2"].astype("category")

        df_equals(ref, exp)

    def test_h2o_q3(self):
        df = self._get_h2o_df()

        ref = df.groupby(["id3"], observed=True).agg({"v1": "sum", "v3": "mean"})
        ref.reset_index(inplace=True)

        modin_df = mpd.DataFrame(df)
        modin_df = modin_df.groupby(["id3"], observed=True, as_index=False).agg(
            {"v1": "sum", "v3": "mean"}
        )

        exp = to_pandas(modin_df)
        exp["id3"] = exp["id3"].astype("category")

        df_equals(ref, exp)

    def test_h2o_q4(self):
        df = self._get_h2o_df()

        ref = df.groupby(["id4"], observed=True).agg(
            {"v1": "mean", "v2": "mean", "v3": "mean"}
        )
        ref.reset_index(inplace=True)

        modin_df = mpd.DataFrame(df)
        modin_df = modin_df.groupby(["id4"], observed=True, as_index=False).agg(
            {"v1": "mean", "v2": "mean", "v3": "mean"}
        )

        exp = to_pandas(modin_df)

        df_equals(ref, exp)

    def test_h2o_q5(self):
        df = self._get_h2o_df()

        ref = df.groupby(["id6"], observed=True).agg(
            {"v1": "sum", "v2": "sum", "v3": "sum"}
        )
        ref.reset_index(inplace=True)

        modin_df = mpd.DataFrame(df)
        modin_df = modin_df.groupby(["id6"], observed=True, as_index=False).agg(
            {"v1": "sum", "v2": "sum", "v3": "sum"}
        )

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

        modin_df = mpd.DataFrame(df)
        modin_df = modin_df.groupby(["id3"], observed=True).agg(
            {"v1": "max", "v2": "min"}
        )
        modin_df["range_v1_v2"] = modin_df["v1"] - modin_df["v2"]
        modin_df = modin_df[["range_v1_v2"]]
        modin_df.reset_index(inplace=True)

        exp = to_pandas(modin_df)
        exp["id3"] = exp["id3"].astype("category")

        df_equals(ref, exp)

    def test_h2o_q10(self):
        df = self._get_h2o_df()

        ref = df.groupby(["id1", "id2", "id3", "id4", "id5", "id6"], observed=True).agg(
            {"v3": "sum", "v1": "count"}
        )
        ref.reset_index(inplace=True)

        modin_df = mpd.DataFrame(df)
        modin_df = modin_df.groupby(
            ["id1", "id2", "id3", "id4", "id5", "id6"], observed=True
        ).agg({"v3": "sum", "v1": "count"})
        modin_df.reset_index(inplace=True)

        exp = to_pandas(modin_df)
        exp["id1"] = exp["id1"].astype("category")
        exp["id2"] = exp["id2"].astype("category")
        exp["id3"] = exp["id3"].astype("category")

        df_equals(ref, exp)


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
        df = pd.DataFrame(data)
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

        modin_lhs = mpd.DataFrame(lhs)
        modin_rhs = mpd.DataFrame(rhs)
        modin_res = modin_lhs.merge(modin_rhs, on="id1")

        exp = to_pandas(modin_res)
        self._fix_category_cols(exp)

        df_equals(ref, exp)

    def test_h2o_q2(self):
        lhs = self._get_h2o_df(self.h2o_data)
        rhs = self._get_h2o_df(self.h2o_data_medium)

        ref = lhs.merge(rhs, on="id2")
        self._fix_category_cols(ref)

        modin_lhs = mpd.DataFrame(lhs)
        modin_rhs = mpd.DataFrame(rhs)
        modin_res = modin_lhs.merge(modin_rhs, on="id2")

        exp = to_pandas(modin_res)
        self._fix_category_cols(exp)

        df_equals(ref, exp)

    def test_h2o_q3(self):
        lhs = self._get_h2o_df(self.h2o_data)
        rhs = self._get_h2o_df(self.h2o_data_medium)

        ref = lhs.merge(rhs, how="left", on="id2")
        self._fix_category_cols(ref)

        modin_lhs = mpd.DataFrame(lhs)
        modin_rhs = mpd.DataFrame(rhs)
        modin_res = modin_lhs.merge(modin_rhs, how="left", on="id2")

        exp = to_pandas(modin_res)
        self._fix_category_cols(exp)

        df_equals(ref, exp)

    def test_h2o_q4(self):
        lhs = self._get_h2o_df(self.h2o_data)
        rhs = self._get_h2o_df(self.h2o_data_medium)

        ref = lhs.merge(rhs, on="id5")
        self._fix_category_cols(ref)

        modin_lhs = mpd.DataFrame(lhs)
        modin_rhs = mpd.DataFrame(rhs)
        modin_res = modin_lhs.merge(modin_rhs, on="id5")

        exp = to_pandas(modin_res)
        self._fix_category_cols(exp)

        df_equals(ref, exp)

    def test_h2o_q5(self):
        lhs = self._get_h2o_df(self.h2o_data)
        rhs = self._get_h2o_df(self.h2o_data_big)

        ref = lhs.merge(rhs, on="id3")
        self._fix_category_cols(ref)

        modin_lhs = mpd.DataFrame(lhs)
        modin_rhs = mpd.DataFrame(rhs)
        modin_res = modin_lhs.merge(modin_rhs, on="id3")

        exp = to_pandas(modin_res)
        self._fix_category_cols(exp)

        df_equals(ref, exp)


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
        "c": [50.0, 40.0, 30.0, 20.0, 10.0],
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
            return getattr(df, cmp_fn)([3, 30, 30.0])

        run_and_compare(cmp, data=self.cmp_data, cmp_fn=cmp_fn)

    @pytest.mark.parametrize("cmp_fn", cmp_fn_values)
    def test_cmp_cols(self, cmp_fn):
        def cmp1(df, cmp_fn, **kwargs):
            return getattr(df["b"], cmp_fn)(df["c"])

        def cmp2(df, cmp_fn, **kwargs):
            return getattr(df[["b", "c"]], cmp_fn)(df[["a", "b"]])

        run_and_compare(cmp1, data=self.cmp_data, cmp_fn=cmp_fn)
        run_and_compare(cmp2, data=self.cmp_data, cmp_fn=cmp_fn)


class TestDateTime:
    datetime_data = {
        "a": [1, 1, 2, 2],
        "b": [11, 21, 12, 11],
        "c": pd.to_datetime(
            ["20190902", "20180913", "20190921", "20180903"], format="%Y%m%d"
        ),
    }

    def test_dt_year(self):
        df = pd.DataFrame(self.datetime_data)
        ref = df["c"].dt.year

        modin_df = mpd.DataFrame(self.datetime_data)
        modin_df = modin_df["c"].dt.year

        exp = to_pandas(modin_df)

        df_equals(ref, exp)

    def test_dt_month(self):
        df = pd.DataFrame(self.datetime_data)
        ref = df["c"].dt.month

        modin_df = mpd.DataFrame(self.datetime_data)
        modin_df = modin_df["c"].dt.month

        exp = to_pandas(modin_df)

        df_equals(ref, exp)


class TestCategory:
    data = {
        "a": ["str1", "str2", "str1", "str3", "str2"],
    }

    def test_cat_codes(self):
        pandas_df = pd.DataFrame(self.data)
        pandas_df["a"] = pandas_df["a"].astype("category")

        modin_df = mpd.DataFrame(pandas_df)

        modin_df["a"] = modin_df["a"].cat.codes
        exp = to_pandas(modin_df)

        pandas_df["a"] = pandas_df["a"].cat.codes

        df_equals(pandas_df, exp)
