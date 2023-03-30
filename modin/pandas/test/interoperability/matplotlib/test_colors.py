from numpy.testing import assert_array_equal

import matplotlib.colors as mcolors

import modin.pandas as pd


def test_pandas_iterable():
    # Using a list or series yields equivalent
    # colormaps, i.e the series isn't seen as
    # a single color
    lst = ["red", "blue", "green"]
    s = pd.Series(lst)
    cm1 = mcolors.ListedColormap(lst, N=5)
    cm2 = mcolors.ListedColormap(s, N=5)
    assert_array_equal(cm1.colors, cm2.colors)
