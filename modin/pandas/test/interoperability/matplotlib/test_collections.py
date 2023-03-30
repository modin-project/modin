import modin.pandas as pd
from matplotlib.collections import Collection


def test_pandas_indexing():

    # Should not fail break when faced with a
    # non-zero indexed series
    index = [11, 12, 13]
    ec = fc = pd.Series(["red", "blue", "green"], index=index)
    lw = pd.Series([1, 2, 3], index=index)
    ls = pd.Series(["solid", "dashed", "dashdot"], index=index)
    aa = pd.Series([True, False, True], index=index)

    Collection(edgecolors=ec)
    Collection(facecolors=fc)
    Collection(linewidths=lw)
    Collection(linestyles=ls)
    Collection(antialiaseds=aa)
