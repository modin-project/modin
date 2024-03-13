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

import inspect

import numpy as np
import pandas
import pytest

import modin.pandas as pd


def test_top_level_api_equality():
    modin_dir = [obj for obj in dir(pd) if obj[0] != "_"]
    pandas_dir = [obj for obj in dir(pandas) if obj[0] != "_"]
    missing_from_modin = set(pandas_dir) - set(modin_dir)
    extra_in_modin = set(modin_dir) - set(pandas_dir)
    ignore_pandas = [
        "annotations",
        "np",
        "tests",
        "pandas",
        "core",
        "compat",
        "util",
        "offsets",
        "datetime",
        "api",
        "tseries",
        "to_msgpack",  # This one is experimental, and doesn't look finished
        "Panel",  # This is deprecated and throws a warning every time.
    ]

    ignore_modin = [
        "indexing",
        "iterator",
        "series",
        "accessor",
        "base",
        "utils",
        "dataframe",
        "groupby",
        "general",
        "datetime",
        "warnings",
        "os",
        "series_utils",
        "window",
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

    ignore_in_pandas = ["timetuple"]
    # modin - namespace for accessing additional Modin functions that are not available in Pandas
    ignore_in_modin = ["modin"]
    missing_from_modin = set(pandas_dir) - set(modin_dir)
    assert not len(
        missing_from_modin - set(ignore_in_pandas)
    ), "Differences found in API: {}".format(
        len(missing_from_modin - set(ignore_in_pandas))
    )
    assert not len(
        set(modin_dir) - set(ignore_in_modin) - set(pandas_dir)
    ), "Differences found in API: {}".format(set(modin_dir) - set(pandas_dir))

    # These have to be checked manually
    allowed_different = ["modin"]

    assert_parameters_eq((pandas.DataFrame, pd.DataFrame), modin_dir, allowed_different)


def test_series_str_api_equality():
    modin_dir = [obj for obj in dir(pd.Series.str) if obj[0] != "_"]
    pandas_dir = [obj for obj in dir(pandas.Series.str) if obj[0] != "_"]

    missing_from_modin = set(pandas_dir) - set(modin_dir)
    assert not len(missing_from_modin), "Differences found in API: {}".format(
        missing_from_modin
    )
    extra_in_modin = set(modin_dir) - set(pandas_dir)
    assert not len(extra_in_modin), "Differences found in API: {}".format(
        extra_in_modin
    )
    assert_parameters_eq((pandas.Series.str, pd.Series.str), modin_dir, [])


def test_series_dt_api_equality():
    modin_dir = [obj for obj in dir(pd.Series.dt) if obj[0] != "_"]
    pandas_dir = [obj for obj in dir(pandas.Series.dt) if obj[0] != "_"]

    # should be deleted, but for some reason the check fails
    # https://github.com/pandas-dev/pandas/pull/33595
    ignore = ["week", "weekofyear"]
    missing_from_modin = set(pandas_dir) - set(modin_dir) - set(ignore)
    assert not len(missing_from_modin), "Differences found in API: {}".format(
        missing_from_modin
    )
    extra_in_modin = set(modin_dir) - set(pandas_dir)
    assert not len(extra_in_modin), "Differences found in API: {}".format(
        extra_in_modin
    )
    assert_parameters_eq((pandas.Series.dt, pd.Series.dt), modin_dir, [])


def test_series_cat_api_equality():
    modin_dir = [obj for obj in dir(pd.Series.cat) if obj[0] != "_"]
    pandas_dir = [obj for obj in dir(pandas.Series.cat) if obj[0] != "_"]

    missing_from_modin = set(pandas_dir) - set(modin_dir)
    assert not len(missing_from_modin), "Differences found in API: {}".format(
        len(missing_from_modin)
    )
    extra_in_modin = set(modin_dir) - set(pandas_dir)
    assert not len(extra_in_modin), "Differences found in API: {}".format(
        extra_in_modin
    )
    # all methods of `pandas.Series.cat` don't have any information about parameters,
    # just method(*args, **kwargs)
    assert_parameters_eq((pandas.core.arrays.Categorical, pd.Series.cat), modin_dir, [])


@pytest.mark.parametrize("obj", ["DataFrame", "Series"])
def test_sparse_accessor_api_equality(obj):
    modin_dir = [x for x in dir(getattr(pd, obj).sparse) if x[0] != "_"]
    pandas_dir = [x for x in dir(getattr(pandas, obj).sparse) if x[0] != "_"]

    missing_from_modin = set(pandas_dir) - set(modin_dir)
    assert not len(missing_from_modin), "Differences found in API: {}".format(
        len(missing_from_modin)
    )
    extra_in_modin = set(modin_dir) - set(pandas_dir)
    assert not len(extra_in_modin), "Differences found in API: {}".format(
        extra_in_modin
    )


@pytest.mark.parametrize("obj", ["SeriesGroupBy", "DataFrameGroupBy"])
def test_groupby_api_equality(obj):
    modin_dir = [x for x in dir(getattr(pd.groupby, obj)) if x[0] != "_"]
    pandas_dir = [x for x in dir(getattr(pandas.core.groupby, obj)) if x[0] != "_"]
    # These attributes are not mentioned in the pandas documentation,
    # but we might want to implement them someday.
    ignore = ["keys", "level", "grouper"]
    missing_from_modin = set(pandas_dir) - set(modin_dir) - set(ignore)
    assert not len(missing_from_modin), "Differences found in API: {}".format(
        len(missing_from_modin)
    )
    # FIXME: wrong inheritance
    ignore = (
        ["boxplot", "corrwith", "dtypes"] if obj == "SeriesGroupBy" else ["boxplot"]
    )
    extra_in_modin = set(modin_dir) - set(pandas_dir) - set(ignore)
    assert not len(extra_in_modin), "Differences found in API: {}".format(
        extra_in_modin
    )
    assert_parameters_eq(
        (getattr(pandas.core.groupby, obj), getattr(pd.groupby, obj)), modin_dir, ignore
    )


def test_series_api_equality():
    modin_dir = [obj for obj in dir(pd.Series) if obj[0] != "_"]
    pandas_dir = [obj for obj in dir(pandas.Series) if obj[0] != "_"]

    ignore = ["timetuple"]
    missing_from_modin = set(pandas_dir) - set(modin_dir) - set(ignore)
    assert not len(missing_from_modin), "Differences found in API: {}".format(
        missing_from_modin
    )
    # modin - namespace for accessing additional Modin functions that are not available in Pandas
    ignore_in_modin = ["modin"]
    extra_in_modin = set(modin_dir) - set(ignore_in_modin) - set(pandas_dir)
    assert not len(extra_in_modin), "Differences found in API: {}".format(
        extra_in_modin
    )

    # These have to be checked manually
    allowed_different = ["modin"]

    assert_parameters_eq((pandas.Series, pd.Series), modin_dir, allowed_different)


def assert_parameters_eq(objects, attributes, allowed_different):
    pandas_obj, modin_obj = objects
    difference = []

    # Check that Modin functions/methods don't have extra params
    for m in attributes:
        if m in allowed_different:
            continue
        try:
            pandas_sig = dict(inspect.signature(getattr(pandas_obj, m)).parameters)
        except TypeError:
            continue
        try:
            modin_sig = dict(inspect.signature(getattr(modin_obj, m)).parameters)
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

    difference = []
    # Check that Modin functions/methods have all params as pandas
    for m in attributes:
        if m in allowed_different:
            continue
        try:
            pandas_sig = dict(inspect.signature(getattr(pandas_obj, m)).parameters)
        except TypeError:
            continue
        try:
            modin_sig = dict(inspect.signature(getattr(modin_obj, m)).parameters)
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
