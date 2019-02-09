import modin.pandas as pd
import pandas
import inspect


def test_api_equality():
    modin_dir = [obj for obj in dir(pd.DataFrame) if obj[0] != "_"]
    pandas_dir = [obj for obj in dir(pandas.DataFrame) if obj[0] != "_"]

    ignore = ["timetuple"]
    missing_from_modin = set(pandas_dir) - set(modin_dir)
    assert not len(missing_from_modin - set(ignore))

    assert not len(set(modin_dir) - set(pandas_dir))

    # These have to be checked manually
    allowed_different = ["to_hdf", "hist"]
    difference = []

    for m in modin_dir:
        if m in allowed_different:
            continue
        try:
            pandas_sig = list(
                inspect.signature(getattr(pandas.DataFrame, m)).parameters.keys()
            )
        except TypeError:
            continue
        try:
            modin_sig = list(
                inspect.signature(getattr(pd.DataFrame, m)).parameters.keys()
            )
        except TypeError:
            continue

        if not pandas_sig == modin_sig:
            difference.append(m)

    assert not len(difference), "Differences found in API: {}".format(difference)
