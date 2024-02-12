import pytest

from modin.dataframe_api_standard.tests.utils import (
    BaseHandler,
    compare_dataframe_with_reference,
    integer_dataframe_4,
)


def test_aggregate(library: BaseHandler) -> None:
    if library.name == "modin":
        pytest.skip("https://github.com/modin-project/modin/issues/3602")
    df = integer_dataframe_4(library)
    df = df.assign((df.col("b") > 0).rename("d"))
    ns = df.__dataframe_namespace__()
    result = (
        df.group_by("key")
        .aggregate(
            ns.Aggregation.sum("b").rename("b_sum"),
            ns.Aggregation.prod("b").rename("b_prod"),
            ns.Aggregation.mean("b").rename("b_mean"),
            ns.Aggregation.median("b").rename("b_median"),
            ns.Aggregation.min("b").rename("b_min"),
            ns.Aggregation.max("b").rename("b_max"),
            ns.Aggregation.std("b").rename("b_std"),
            ns.Aggregation.var("b").rename("b_var"),
            ns.Aggregation.size().rename("b_count"),
            ns.Aggregation.any("d").rename("d_any"),
            ns.Aggregation.all("d").rename("d_all"),
        )
        .sort("key")
    )
    expected = {
        "key": [1, 2],
        "b_sum": [3, 7],
        "b_prod": [2, 12],
        "b_mean": [1.5, 3.5],
        "b_median": [1.5, 3.5],
        "b_min": [1, 3],
        "b_max": [2, 4],
        "b_std": [0.707107, 0.707107],
        "b_var": [0.5, 0.5],
        "b_count": [2, 2],
        "d_any": [True, True],
        "d_all": [True, True],
    }
    expected_dtype = {
        "key": ns.Int64,
        "b_sum": ns.Int64,
        "b_prod": ns.Int64,
        "b_mean": ns.Float64,
        "b_median": ns.Float64,
        "b_min": ns.Int64,
        "b_max": ns.Int64,
        "b_std": ns.Float64,
        "b_var": ns.Float64,
        "b_count": ns.Int64,
        "d_any": ns.Bool,
        "d_all": ns.Bool,
    }
    compare_dataframe_with_reference(result, expected, dtype=expected_dtype)  # type: ignore[arg-type]


def test_aggregate_only_size(library: BaseHandler) -> None:
    if library.name == "modin":
        pytest.skip("https://github.com/modin-project/modin/issues/3602")
    df = integer_dataframe_4(library)
    ns = df.__dataframe_namespace__()
    result = (
        df.group_by("key")
        .aggregate(
            ns.Aggregation.size().rename("b_count"),
        )
        .sort("key")
    )
    expected = {
        "key": [1, 2],
        "b_count": [2, 2],
    }
    compare_dataframe_with_reference(result, expected, dtype=ns.Int64)


def test_aggregate_no_size(library: BaseHandler) -> None:
    df = integer_dataframe_4(library)
    ns = df.__dataframe_namespace__()
    result = (
        df.group_by("key")
        .aggregate(
            ns.Aggregation.sum("b").rename("b_sum"),
            ns.Aggregation.mean("b").rename("b_mean"),
            ns.Aggregation.min("b").rename("b_min"),
            ns.Aggregation.max("b").rename("b_max"),
            ns.Aggregation.std("b").rename("b_std"),
            ns.Aggregation.var("b").rename("b_var"),
        )
        .sort("key")
    )
    expected = {
        "key": [1, 2],
        "b_sum": [3, 7],
        "b_mean": [1.5, 3.5],
        "b_min": [1, 3],
        "b_max": [2, 4],
        "b_std": [0.707107, 0.707107],
        "b_var": [0.5, 0.5],
    }
    expected_dtype = {
        "key": ns.Int64,
        "b_sum": ns.Int64,
        "b_mean": ns.Float64,
        "b_min": ns.Int64,
        "b_max": ns.Int64,
        "b_std": ns.Float64,
        "b_var": ns.Float64,
    }
    compare_dataframe_with_reference(result, expected, dtype=expected_dtype)  # type: ignore[arg-type]
