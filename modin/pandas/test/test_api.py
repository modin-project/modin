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

import modin.pandas as pd
import pandas
import inspect
import numpy as np


def test_top_level_api_equality():
    modin_dir = [obj for obj in dir(pd) if obj[0] != "_"]
    pandas_dir = [obj for obj in dir(pandas) if obj[0] != "_"]
    missing_from_modin = set(pandas_dir) - set(modin_dir)
    extra_in_modin = set(modin_dir) - set(pandas_dir)
    ignore_pandas = [
        "np",
        "testing",
        "tests",
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
        "Panel",  # This is deprecated and throws a warning every time.
    ]

    ignore_modin = [
        "DEFAULT_NPARTITIONS",
        "iterator",
        "series",
        "accessor",
        "base",
        "utils",
        "dataframe",
        "groupby",
        "threading",
        "general",
        "datetimes",
        "reshape",
        "types",
        "sys",
        "initialize_ray",
        "datetime",
        "ray",
        "num_cpus",
        "warnings",
        "os",
        "multiprocessing",
        "Client",
        "dask_client",
        "get_client",
    ]

    assert not len(
        missing_from_modin - set(ignore_pandas)
    ), "Differences found in API: {}".format(missing_from_modin - set(ignore_pandas))

    assert not len(
        extra_in_modin - set(ignore_modin)
    ), "Differences found in API: {}".format(extra_in_modin - set(ignore_modin))

    difference = []
    allowed_different = ["Interval", "datetime"]

    # Check that we have all keywords and defaults in pandas
    for m in set(pandas_dir) - set(ignore_pandas):
        if m in allowed_different:
            continue
        try:
            pandas_sig = dict(inspect.signature(getattr(pandas, m)).parameters)
        except (TypeError, ValueError):
            continue
        try:
            modin_sig = dict(inspect.signature(getattr(pd, m)).parameters)
        except (TypeError, ValueError):
            continue

        if not pandas_sig == modin_sig:
            try:
                append_val = (
                    m,
                    {
                        i: pandas_sig[i]
                        for i in pandas_sig.keys()
                        if i not in modin_sig
                        or pandas_sig[i].default != modin_sig[i].default
                        and not (
                            pandas_sig[i].default is np.nan
                            and modin_sig[i].default is np.nan
                        )
                    },
                )
            except Exception:
                raise
            try:
                # This validates that there are actually values to add to the difference
                # based on the condition above.
                if len(list(append_val[-1])[-1]) > 0:
                    difference.append(append_val)
            except IndexError:
                pass

    assert not len(difference), "Missing params found in API: {}".format(difference)

    # Check that we have no extra keywords or defaults
    for m in set(pandas_dir) - set(ignore_pandas):
        if m in allowed_different:
            continue
        try:
            pandas_sig = dict(inspect.signature(getattr(pandas, m)).parameters)
        except (TypeError, ValueError):
            continue
        try:
            modin_sig = dict(inspect.signature(getattr(pd, m)).parameters)
        except (TypeError, ValueError):
            continue
        if not pandas_sig == modin_sig:
            try:
                append_val = (
                    m,
                    {
                        i: modin_sig[i]
                        for i in modin_sig.keys()
                        if i not in pandas_sig and i != "query_compiler"
                    },
                )
            except Exception:
                raise
            try:
                # This validates that there are actually values to add to the difference
                # based on the condition above.
                if len(list(append_val[-1])[-1]) > 0:
                    difference.append(append_val)
            except IndexError:
                pass

    assert not len(difference), "Extra params found in API: {}".format(difference)


def test_dataframe_api_equality():
    modin_dir = [obj for obj in dir(pd.DataFrame) if obj[0] != "_"]
    pandas_dir = [obj for obj in dir(pandas.DataFrame) if obj[0] != "_"]

    ignore = ["timetuple"]
    missing_from_modin = set(pandas_dir) - set(modin_dir)
    assert not len(
        missing_from_modin - set(ignore)
    ), "Differences found in API: {}".format(len(missing_from_modin - set(ignore)))
    assert not len(
        set(modin_dir) - set(pandas_dir)
    ), "Differences found in API: {}".format(set(modin_dir) - set(pandas_dir))

    # These have to be checked manually
    allowed_different = ["to_hdf", "hist"]
    difference = []

    # Check that we don't have extra params
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
                    if i not in modin_sig
                    or pandas_sig[i].default != modin_sig[i].default
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

    assert not len(difference), "Missing params found in API: {}".format(difference)

    # Check that we have all params
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
                {i: modin_sig[i] for i in modin_sig.keys() if i not in pandas_sig},
            )
            try:
                # This validates that there are actually values to add to the difference
                # based on the condition above.
                if len(list(append_val[-1])[-1]) > 0:
                    difference.append(append_val)
            except IndexError:
                pass

    assert not len(difference), "Extra params found in API: {}".format(difference)


def test_series_api_equality():
    modin_dir = [obj for obj in dir(pd.Series) if obj[0] != "_"]
    pandas_dir = [obj for obj in dir(pandas.Series) if obj[0] != "_"]

    ignore = ["timetuple"]
    missing_from_modin = set(pandas_dir) - set(modin_dir)
    assert not len(
        missing_from_modin - set(ignore)
    ), "Differences found in API: {}".format(len(missing_from_modin - set(ignore)))
    assert not len(
        set(modin_dir) - set(pandas_dir)
    ), "Differences found in API: {}".format(set(modin_dir) - set(pandas_dir))

    # These have to be checked manually
    allowed_different = ["to_hdf", "hist"]
    difference = []

    for m in modin_dir:
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
                    if i not in modin_sig
                    or pandas_sig[i].default != modin_sig[i].default
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
    assert not len(difference), "Missing params found in API: {}".format(difference)

    for m in modin_dir:
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
                {i: modin_sig[i] for i in modin_sig.keys() if i not in pandas_sig},
            )
            try:
                # This validates that there are actually values to add to the difference
                # based on the condition above.
                if len(list(append_val[-1])[-1]) > 0:
                    difference.append(append_val)
            except IndexError:
                pass
    assert not len(difference), "Extra params found in API: {}".format(difference)
