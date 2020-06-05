import os

os.environ["MODIN_EXPERIMENTAL"] = "True"
os.environ["MODIN_ENGINE"] = "ray"
os.environ["MODIN_BACKEND"] = "omnisci"

if os.environ["OMNISCI_SERVER"] is None:
    raise RuntimeError("OMNISCI_SERVER variable must be initialized")

import modin.pandas as mpd
import pandas as pd

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
        modin_df = pd.DataFrame(self.data)
        pandas_df["d"] = pandas_df["a"]
        modin_df["d"] = modin_df["a"]
        df_equals(pandas_df, modin_df)


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
    on_values = ["a"]
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
