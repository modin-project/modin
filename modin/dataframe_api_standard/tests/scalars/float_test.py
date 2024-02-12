import numpy as np
import pytest

from modin.dataframe_api_standard.tests.utils import (
    BaseHandler,
    compare_dataframe_with_reference,
    integer_dataframe_1,
    integer_dataframe_2,
)


@pytest.mark.parametrize(
    "attr",
    [
        "__lt__",
        "__le__",
        "__eq__",
        "__ne__",
        "__gt__",
        "__ge__",
        "__add__",
        "__radd__",
        "__sub__",
        "__rsub__",
        "__mul__",
        "__rmul__",
        "__mod__",
        "__rmod__",
        "__pow__",
        "__rpow__",
        "__floordiv__",
        "__rfloordiv__",
        "__truediv__",
        "__rtruediv__",
    ],
)
def test_float_binary(library: BaseHandler, attr: str) -> None:
    other = 0.5
    df = integer_dataframe_2(library).persist()
    scalar = df.col("a").mean()
    float_scalar = float(scalar)  # type: ignore[arg-type]
    assert getattr(scalar, attr)(other) == getattr(float_scalar, attr)(other)


def test_float_binary_invalid(library: BaseHandler) -> None:
    lhs = integer_dataframe_2(library).col("a").mean()
    rhs = integer_dataframe_1(library).col("b").mean()
    with pytest.raises(ValueError):
        _ = lhs > rhs


def test_float_binary_lazy_valid(library: BaseHandler) -> None:
    df = integer_dataframe_2(library).persist()
    lhs = df.col("a").mean()
    rhs = df.col("b").mean()
    result = lhs > rhs
    assert not bool(result)


@pytest.mark.parametrize(
    "attr",
    [
        "__abs__",
        "__neg__",
    ],
)
def test_float_unary(library: BaseHandler, attr: str) -> None:
    df = integer_dataframe_2(library).persist()
    with pytest.warns(UserWarning):
        scalar = df.col("a").persist().mean()
    float_scalar = float(scalar)  # type: ignore[arg-type]
    assert getattr(scalar, attr)() == getattr(float_scalar, attr)()


@pytest.mark.parametrize(
    "attr",
    [
        "__int__",
        "__float__",
        "__bool__",
    ],
)
def test_float_unary_invalid(library: BaseHandler, attr: str) -> None:
    df = integer_dataframe_2(library)
    scalar = df.col("a").mean()
    float_scalar = float(scalar.persist())  # type: ignore[arg-type]
    with pytest.raises(RuntimeError):
        assert getattr(scalar, attr)() == getattr(float_scalar, attr)()


def test_free_standing(library: BaseHandler) -> None:
    df = integer_dataframe_1(library)
    namespace = df.__dataframe_namespace__()
    ser = namespace.column_from_1d_array(  # type: ignore[call-arg]
        np.array([1, 2, 3]),
        name="a",
    )
    result = float(ser.mean() + 1)  # type: ignore[arg-type]
    assert result == 3.0


def test_right_comparand(library: BaseHandler) -> None:
    df = integer_dataframe_1(library)
    ns = df.__dataframe_namespace__()
    col = df.col("a")  # [1, 2, 3]
    scalar = df.col("b").get_value(0)  # 4
    result = df.assign((scalar - col).rename("c"))
    expected = {
        "a": [1, 2, 3],
        "b": [4, 5, 6],
        "c": [3, 2, 1],
    }
    compare_dataframe_with_reference(result, expected, ns.Int64)
