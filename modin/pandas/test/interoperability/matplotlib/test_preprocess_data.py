import numpy as np
import pytest

from matplotlib import _preprocess_data
import modin.pandas as pd

# Notes on testing the plotting functions itself
# *   the individual decorated plotting functions are tested in 'test_axes.py'
# *   that pyplot functions accept a data kwarg is only tested in
#     test_axes.test_pie_linewidth_0


# this gets used in multiple tests, so define it here
@_preprocess_data(replace_names=["x", "y"], label_namer="y")
def plot_func(ax, x, y, ls="x", label=None, w="xyz"):
    return "x: %s, y: %s, ls: %s, w: %s, label: %s" % (list(x), list(y), ls, w, label)


all_funcs = [plot_func]
all_func_ids = ["plot_func"]


@pytest.mark.parametrize("func", all_funcs, ids=all_func_ids)
def test_function_call_with_pandas_data(func):
    """Test with pandas dataframe -> label comes from ``data["col"].name``."""
    data = pd.DataFrame(
        {
            "a": np.array([1, 2], dtype=np.int32),
            "b": np.array([8, 9], dtype=np.int32),
            "w": ["NOT", "NOT"],
        }
    )

    assert (
        func(None, "a", "b", data=data)
        == "x: [1, 2], y: [8, 9], ls: x, w: xyz, label: b"
    )
    assert (
        func(None, x="a", y="b", data=data)
        == "x: [1, 2], y: [8, 9], ls: x, w: xyz, label: b"
    )
    assert (
        func(None, "a", "b", label="", data=data)
        == "x: [1, 2], y: [8, 9], ls: x, w: xyz, label: "
    )
    assert (
        func(None, "a", "b", label="text", data=data)
        == "x: [1, 2], y: [8, 9], ls: x, w: xyz, label: text"
    )
    assert (
        func(None, x="a", y="b", label="", data=data)
        == "x: [1, 2], y: [8, 9], ls: x, w: xyz, label: "
    )
    assert (
        func(None, x="a", y="b", label="text", data=data)
        == "x: [1, 2], y: [8, 9], ls: x, w: xyz, label: text"
    )
