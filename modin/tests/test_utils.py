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

import contextlib
import json
from textwrap import dedent, indent
from unittest.mock import Mock, patch

import numpy as np
import pandas
import pytest

import modin.pandas as pd
import modin.utils
from modin.config import NativeDataframeMode
from modin.error_message import ErrorMessage
from modin.tests.pandas.utils import create_test_dfs


# Note: classes below are used for purely testing purposes - they
# simulate real-world use cases for _inherit_docstring
class BaseParent:
    def method(self):
        """ordinary method (base)"""

    def base_method(self):
        """ordinary method in base only"""

    @property
    def prop(self):
        """property"""

    @staticmethod
    def static():
        """static method"""

    @classmethod
    def clsmtd(cls):
        """class method"""


class BaseChild(BaseParent):
    """this is class docstring"""

    def method(self):
        """ordinary method (child)"""

    def own_method(self):
        """own method"""

    def no_overwrite(self):
        """another own method"""

    F = property(method)


@pytest.fixture(scope="module")
def wrapped_cls():
    @modin.utils._inherit_docstrings(BaseChild)
    class Wrapped:
        def method(self):
            pass

        def base_method(self):
            pass

        def own_method(self):
            pass

        def no_overwrite(self):
            """not overwritten doc"""

        @property
        def prop(self):
            return None

        @staticmethod
        def static():
            pass

        @classmethod
        def clsmtd(cls):
            pass

        F = property(method)

    return Wrapped


def _check_doc(wrapped, orig):
    assert wrapped.__doc__ == orig.__doc__
    if isinstance(wrapped, property):
        assert wrapped.fget.__doc_inherited__
    else:
        assert wrapped.__doc_inherited__


def test_doc_inherit_clslevel(wrapped_cls):
    _check_doc(wrapped_cls, BaseChild)


def test_doc_inherit_methods(wrapped_cls):
    _check_doc(wrapped_cls.method, BaseChild.method)
    _check_doc(wrapped_cls.base_method, BaseParent.base_method)
    _check_doc(wrapped_cls.own_method, BaseChild.own_method)
    assert wrapped_cls.no_overwrite.__doc__ != BaseChild.no_overwrite.__doc__
    assert not getattr(wrapped_cls.no_overwrite, "__doc_inherited__", False)


def test_doc_inherit_special(wrapped_cls):
    _check_doc(wrapped_cls.static, BaseChild.static)
    _check_doc(wrapped_cls.clsmtd, BaseChild.clsmtd)


def test_doc_inherit_props(wrapped_cls):
    assert type(wrapped_cls.method) == type(BaseChild.method)  # noqa: E721
    _check_doc(wrapped_cls.prop, BaseChild.prop)
    _check_doc(wrapped_cls.F, BaseChild.F)


def test_doc_inherit_prop_builder():
    def builder(name):
        return property(lambda self: name)

    class Parent:
        prop = builder("Parent")

    @modin.utils._inherit_docstrings(Parent)
    class Child(Parent):
        prop = builder("Child")

    assert Parent().prop == "Parent"
    assert Child().prop == "Child"


@pytest.mark.parametrize(
    "source_doc,to_append,expected",
    [
        (
            "One-line doc.",
            "One-line message.",
            "One-line doc.One-line message.",
        ),
        (
            """
            Regular doc-string
                With the setted indent style.
            """,
            """
                    Doc-string having different indents
                        in comparison with the regular one.
            """,
            """
            Regular doc-string
                With the setted indent style.

            Doc-string having different indents
                in comparison with the regular one.
            """,
        ),
    ],
)
def test_append_to_docstring(source_doc, to_append, expected):
    def source_fn():
        pass

    source_fn.__doc__ = source_doc
    result_fn = modin.utils.append_to_docstring(to_append)(source_fn)

    answer = dedent(result_fn.__doc__)
    expected = dedent(expected)

    assert answer == expected


def test_align_indents():
    source = """
    Source string that sets
        the indent pattern."""
    target = indent(source, " " * 5)
    result = modin.utils.align_indents(source, target)
    assert source == result


def test_format_string():
    template = """
            Source template string that has some {inline_placeholder}s.
            Placeholder1:
            {new_line_placeholder1}
            Placeholder2:
            {new_line_placeholder2}
            Placeholder3:
            {new_line_placeholder3}
            Placeholder4:
            {new_line_placeholder4}Text text:
                Placeholder5:
                {new_line_placeholder5}
    """

    singleline_value = "Single-line value"
    multiline_value = """
        Some string
            Having different indentation
        From the source one."""
    multiline_value_new_line_at_the_end = multiline_value + "\n"
    multiline_value_new_line_at_the_begin = "\n" + multiline_value

    expected = """
            Source template string that has some Single-line values.
            Placeholder1:
            Some string
                Having different indentation
            From the source one.
            Placeholder2:
            Single-line value
            Placeholder3:
            
            Some string
                Having different indentation
            From the source one.
            Placeholder4:
            Some string
                Having different indentation
            From the source one.
            Text text:
                Placeholder5:
                Some string
                    Having different indentation
                From the source one.
    """  # noqa: W293
    answer = modin.utils.format_string(
        template,
        inline_placeholder=singleline_value,
        new_line_placeholder1=multiline_value,
        new_line_placeholder2=singleline_value,
        new_line_placeholder3=multiline_value_new_line_at_the_begin,
        new_line_placeholder4=multiline_value_new_line_at_the_end,
        new_line_placeholder5=multiline_value,
    )
    assert answer == expected


def warns_that_defaulting_to_pandas(prefix=None, suffix=None, force=False):
    """
    Assert that code warns that it's defaulting to pandas.

    Parameters
    ----------
    prefix : Optional[str]
        If specified, checks that the start of the warning message matches this argument
        before "[Dd]efaulting to pandas".
    suffix : Optional[str]
        If specified, checks that the end of the warning message matches this argument
        after "[Dd]efaulting to pandas".
    force : Optional[bool]
        If ``True``, return the ``pytest.recwarn.WarningsChecker`` irrespective of ``NativeDataframeMode``.

    Returns
    -------
    pytest.recwarn.WarningsChecker or contextlib.nullcontext
        If Modin is not operating in ``NativeDataframeMode``, a ``WarningsChecker``
        is returned, which will check for a ``UserWarning`` indicating that Modin
        is defaulting to Pandas. If ``NativeDataframeMode`` is set, a
        ``nullcontext`` is returned to avoid the warning about defaulting to Pandas,
        as this occurs due to user setting of ``NativeDataframeMode``.
    """
    if NativeDataframeMode.get() == "Pandas" and not force:
        return contextlib.nullcontext()

    match = "[Dd]efaulting to pandas"
    if prefix:
        # Message may be separated by newlines
        match = match + "(.|\\n)+"
    if suffix:
        match += "(.|\\n)+" + suffix
    return pytest.warns(UserWarning, match=match)


@pytest.mark.parametrize("as_json", [True, False])
def test_show_versions(as_json, capsys):
    modin.utils.show_versions(as_json=as_json)
    versions = capsys.readouterr().out
    assert modin.__version__ in versions

    if as_json:
        versions = json.loads(versions)
        assert versions["modin dependencies"]["modin"] == modin.__version__


def test_warns_that_defaulting_to_pandas():
    with warns_that_defaulting_to_pandas():
        ErrorMessage.default_to_pandas()

    with warns_that_defaulting_to_pandas():
        ErrorMessage.default_to_pandas(message="Function name")


def test_assert_dtypes_equal():
    """Verify that `assert_dtypes_equal` from test utils works correctly (raises an error when it has to)."""
    from modin.tests.pandas.utils import assert_dtypes_equal

    # Serieses with equal dtypes
    sr1, sr2 = pd.Series([1.0]), pandas.Series([1.0])
    assert sr1.dtype == sr2.dtype == "float"
    assert_dtypes_equal(sr1, sr2)  # shouldn't raise an error since dtypes are equal

    # Serieses with different dtypes belonging to the same class
    sr1 = sr1.astype("int")
    assert sr1.dtype != sr2.dtype and sr1.dtype == "int"
    assert_dtypes_equal(sr1, sr2)  # shouldn't raise an error since both are numeric

    # Serieses with different dtypes not belonging to the same class
    sr2 = sr2.astype("str")
    assert sr1.dtype != sr2.dtype and sr2.dtype == "object"
    with pytest.raises(AssertionError):
        assert_dtypes_equal(sr1, sr2)

    # Dfs with equal dtypes
    df1, df2 = create_test_dfs({"a": [1], "b": [1.0]})
    assert_dtypes_equal(df1, df2)  # shouldn't raise an error since dtypes are equal

    # Dfs with different dtypes belonging to the same class
    df1 = df1.astype({"a": "float"})
    assert df1.dtypes["a"] != df2.dtypes["a"]
    assert_dtypes_equal(df1, df2)  # shouldn't raise an error since both are numeric

    # Dfs with different dtypes
    df2 = df2.astype("str")
    with pytest.raises(AssertionError):
        assert_dtypes_equal(sr1, sr2)

    # Dfs with categorical dtypes
    df1 = df1.astype("category")
    df2 = df2.astype("category")
    assert_dtypes_equal(df1, df2)  # shouldn't raise an error since both are categorical

    # Dfs with different dtypes (categorical and str)
    df1 = df1.astype({"a": "str"})
    with pytest.raises(AssertionError):
        assert_dtypes_equal(df1, df2)


def test_execute():
    data = np.random.rand(100, 64)
    modin_df, pandas_df = create_test_dfs(data)
    partitions = modin_df._query_compiler._modin_frame._partitions.flatten()
    mgr_cls = modin_df._query_compiler._modin_frame._partition_mgr_cls

    # check modin case
    with patch.object(mgr_cls, "wait_partitions", new=Mock()):
        modin.utils.execute(modin_df)
        mgr_cls.wait_partitions.assert_called_once()
        assert (mgr_cls.wait_partitions.call_args[0] == partitions).all()

    # check pandas case without error
    with patch.object(mgr_cls, "wait_partitions", new=Mock()):
        modin.utils.execute(pandas_df)
        mgr_cls.wait_partitions.assert_not_called()

    with patch.object(mgr_cls, "wait_partitions", new=Mock()):
        modin.utils.execute(modin_df)
        mgr_cls.wait_partitions.assert_called_once()

    # check several modin dataframes
    with patch.object(mgr_cls, "wait_partitions", new=Mock()):
        modin.utils.execute(modin_df, modin_df[modin_df.columns[:4]])
        mgr_cls.wait_partitions.assert_called
        assert mgr_cls.wait_partitions.call_count == 2
