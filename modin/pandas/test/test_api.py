import modin.pandas as pd
import pandas
import inspect
import numpy as np


def test_top_level_api_equality():
    modin_dir = [obj for obj in dir(pd) if obj[0] != "_"]
    pandas_dir = [obj for obj in dir(pandas) if obj[0] != "_"]
    missing_from_modin = set(pandas_dir) - set(modin_dir)
    ignore = [
        "np",
        "testing",
        "pandas",
        "core",
        "compat",
        "util",
        "offsets",
        "datetime",
        "arrays",
        "api",
        "tseries",
        "errors",
        "to_msgpack",  # This one is experimental, and doesn't look finished
        "describe_option",
        "get_option",
        "option_context",
        "reset_option",
    ]

    assert not len(
        missing_from_modin - set(ignore)
    ), "Differences found in API: {}".format(missing_from_modin - set(ignore))

    difference = []
    allowed_different = ["Interval", "datetime"]

    for m in set(pandas_dir) - set(ignore):
        if m in allowed_different:
            continue
        try:
            pandas_sig = dict(inspect.signature(getattr(pandas, m)).parameters)
        except TypeError:
            continue
        try:
            modin_sig = dict(inspect.signature(getattr(pd, m)).parameters)
        except TypeError:
            continue

        if not pandas_sig == modin_sig:
            try:
                append_val = (
                    m,
                    {
                        i: pandas_sig[i]
                        for i in pandas_sig.keys()
                        if pandas_sig[i] != modin_sig[i]
                        and not (
                            pandas_sig[i].default is np.nan
                            and modin_sig[i].default is np.nan
                        )
                    },
                )
            except Exception:
                print(pandas_sig)
                print(modin_sig)
                print(m)
                raise
            try:
                # This validates that there are actually values to add to the difference
                # based on the condition above.
                if len(list(append_val[-1])[-1]) > 0:
                    difference.append(append_val)
            except IndexError:
                pass

    assert not len(difference), "Differences found in API: {}".format(difference)


def test_dataframe_api_equality():
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
            pandas_sig = dict(
                inspect.signature(getattr(pandas.DataFrame, m)).parameters
            )
        except TypeError:
            continue
        try:
            modin_sig = dict(inspect.signature(getattr(pd.DataFrame, m)).parameters)
        except TypeError:
            continue

        if not pandas_sig == modin_sig:
            append_val = (
                m,
                {
                    i: pandas_sig[i]
                    for i in pandas_sig.keys()
                    if pandas_sig[i] != modin_sig[i]
                    and not (
                        pandas_sig[i].default is np.nan
                        and modin_sig[i].default is np.nan
                    )
                },
            )
            try:
                # This validates that there are actually values to add to the difference
                # based on the condition above.
                if len(list(append_val[-1])[-1]) > 0:
                    difference.append(append_val)
            except IndexError:
                pass

    assert not len(difference), "Differences found in API: {}".format(difference)


def test_series_api_equality():
    modin_dir = [obj for obj in dir(pd.Series) if obj[0] != "_"]
    pandas_dir = [obj for obj in dir(pandas.Series) if obj[0] != "_"]

    ignore = ["timetuple"]
    missing_from_modin = set(pandas_dir) - set(modin_dir)
    assert not len(missing_from_modin - set(ignore)), missing_from_modin

    assert not len(set(modin_dir) - set(pandas_dir)), set(modin_dir) - set(pandas_dir)

    # These have to be checked manually
    allowed_different = ["to_hdf", "hist"]
    difference = []

    for m in modin_dir:
        print(m)
        if m in allowed_different:
            continue
        try:
            pandas_sig = dict(inspect.signature(getattr(pandas.Series, m)).parameters)
        except TypeError:
            continue
        try:
            modin_sig = dict(inspect.signature(getattr(pd.Series, m)).parameters)
        except TypeError:
            continue

        if not pandas_sig == modin_sig:
            append_val = (
                m,
                {
                    i: pandas_sig[i]
                    for i in pandas_sig.keys()
                    if pandas_sig[i] != modin_sig[i]
                    and not (
                        pandas_sig[i].default is np.nan
                        and modin_sig[i].default is np.nan
                    )
                },
            )
            try:
                # This validates that there are actually values to add to the difference
                # based on the condition above.
                if len(list(append_val[-1])[-1]) > 0:
                    difference.append(append_val)
            except IndexError:
                pass

    print(difference)
    assert not len(difference), "Differences found in API: {}".format(difference)
