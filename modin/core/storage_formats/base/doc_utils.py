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

"""Module contains decorators for documentation of the query compiler methods."""

from functools import partial

from modin.utils import align_indents, append_to_docstring, format_string

_one_column_warning = """
.. warning::
    This method is supported only by one-column query compilers.
"""

_deprecation_warning = """
.. warning::
    This method duplicates logic of ``{0}`` and will be removed soon.
"""

_refer_to_note = """
Notes
-----
Please refer to ``modin.pandas.{0}`` for more information
about parameters and output format.
"""

add_one_column_warning = append_to_docstring(_one_column_warning)


def add_deprecation_warning(replacement_method):
    """
    Build decorator which appends deprecation warning to the function's docstring.

    Appended warning indicates that the current method duplicates functionality of
    some other method and so is slated to be removed in the future.

    Parameters
    ----------
    replacement_method : str
        Name of the method to use instead of deprecated.

    Returns
    -------
    callable
    """
    message = _deprecation_warning.format(replacement_method)
    return append_to_docstring(message)


def add_refer_to(method):
    """
    Build decorator which appends link to the high-level equivalent method to the function's docstring.

    Parameters
    ----------
    method : str
        Method name in ``modin.pandas`` module to refer to.

    Returns
    -------
    callable
    """
    # FIXME: this would break numpydoc if there already is a `Notes` section
    note = _refer_to_note.format(method)
    return append_to_docstring(note)


def doc_qc_method(
    template,
    params=None,
    refer_to=None,
    refer_to_module_name=None,
    one_column_method=False,
    **kwargs,
):
    """
    Build decorator which adds docstring for query compiler method.

    Parameters
    ----------
    template : str
        Method docstring in the NumPy docstyle format. Must contain {params}
        placeholder.
    params : str, optional
        Method parameters in the NumPy docstyle format to substitute
        in the `template`. `params` string should not include the "Parameters"
        header.
    refer_to : str, optional
        Method name in `refer_to_module_name` module to refer to for more information
        about parameters and output format.
    refer_to_module_name : str, optional
    one_column_method : bool, default: False
        Whether to append note that this method is for one-column
        query compilers only.
    **kwargs : dict
        Values to substitute in the `template`.

    Returns
    -------
    callable
    """
    params_template = """

        Parameters
        ----------
        {params}
        """

    params = format_string(params_template, params=params) if params else ""
    substituted = format_string(template, params=params, refer_to=refer_to, **kwargs)
    if refer_to_module_name:
        refer_to = f"{refer_to_module_name}.{refer_to}"

    def decorator(func):
        func.__doc__ = substituted
        appendix = ""
        if refer_to:
            appendix += _refer_to_note.format(refer_to)
        if one_column_method:
            appendix += _one_column_warning
        if appendix:
            func = append_to_docstring(appendix)(func)
        return func

    return decorator


def doc_binary_method(operation, sign, self_on_right=False, op_type="arithmetic"):
    """
    Build decorator which adds docstring for binary method.

    Parameters
    ----------
    operation : str
        Name of the binary operation.
    sign : str
        Sign which represents specified binary operation.
    self_on_right : bool, default: False
        Whether `self` is the right operand.
    op_type : {"arithmetic", "logical", "comparison"}, default: "arithmetic"
        Type of the binary operation.

    Returns
    -------
    callable
    """
    template = """
    Perform element-wise {operation} (``{verbose}``).

    If axes are not equal, perform frames alignment first.

    Parameters
    ----------
    other : BaseQueryCompiler, scalar or array-like
        Other operand of the binary operation.
    broadcast : bool, default: False
        If `other` is a one-column query compiler, indicates whether it is a Series or not.
        Frames and Series have to be processed differently, however we can't distinguish them
        at the query compiler level, so this parameter is a hint that is passed from a high-level API.
    {extra_params}**kwargs : dict
        Serves the compatibility purpose. Does not affect the result.

    Returns
    -------
    BaseQueryCompiler
        Result of binary operation.
    """

    extra_params = {
        "logical": """
        level : int or label
            In case of MultiIndex match index values on the passed level.
        axis : {{0, 1}}
            Axis to match indices along for 1D `other` (list or QueryCompiler that represents Series).
            0 is for index, when 1 is for columns.
        """,
        "arithmetic": """
        level : int or label
            In case of MultiIndex match index values on the passed level.
        axis : {{0, 1}}
            Axis to match indices along for 1D `other` (list or QueryCompiler that represents Series).
            0 is for index, when 1 is for columns.
        fill_value : float or None
            Value to fill missing elements during frame alignment.
        """,
        "series_comparison": """
        level : int or label
            In case of MultiIndex match index values on the passed level.
        fill_value : float or None
            Value to fill missing elements during frame alignment.
        axis : {{0, 1}}
            Unused. Parameter needed for compatibility with DataFrame.
        """,
    }

    verbose_substitution = (
        f"other {sign} self" if self_on_right else f"self {sign} other"
    )
    params_substitution = extra_params.get(op_type, "")
    return doc_qc_method(
        template,
        extra_params=params_substitution,
        operation=operation,
        verbose=verbose_substitution,
    )


def doc_reduce_agg(method, refer_to, params=None, extra_params=None):
    """
    Build decorator which adds docstring for the reduce method.

    Parameters
    ----------
    method : str
        The result of the method.
    refer_to : str
        Method name in ``modin.pandas.DataFrame`` module to refer to for
        more information about parameters and output format.
    params : str, optional
        Method parameters in the NumPy docstyle format to substitute
        to the docstring template.
    extra_params : sequence of str, optional
        Method parameter names to append to the docstring template. Parameter
        type and description will be grabbed from ``extra_params_map`` (Please
        refer to the source code of this function to explore the map).

    Returns
    -------
    callable
    """
    template = """
        Get the {method} for each column or row.
        {params}
        Returns
        -------
        BaseQueryCompiler
            One-column QueryCompiler with index labels of the specified axis,
            where each row contains the {method} for the corresponding
            row or column.
        """

    if params is None:
        params = """
        axis : {{0, 1}}
        numeric_only : bool, optional"""

    extra_params_map = {
        "skipna": """
        skipna : bool, default: True""",
        "min_count": """
        min_count : int""",
        "ddof": """
        ddof : int""",
        "*args": """
        *args : iterable
            Serves the compatibility purpose. Does not affect the result.""",
        "**kwargs": """
        **kwargs : dict
            Serves the compatibility purpose. Does not affect the result.""",
    }

    params += "".join(
        [
            align_indents(
                source=params, target=extra_params_map.get(param, f"\n{param} : object")
            )
            for param in (extra_params or [])
        ]
    )
    return doc_qc_method(
        template,
        params=params,
        method=method,
        refer_to=f"DataFrame.{refer_to}",
    )


doc_cum_agg = partial(
    doc_qc_method,
    template="""
    Get cumulative {method} for every row or column.

    Parameters
    ----------
    fold_axis : {{0, 1}}
    skipna : bool
    **kwargs : dict
        Serves the compatibility purpose. Does not affect the result.

    Returns
    -------
    BaseQueryCompiler
        QueryCompiler of the same shape as `self`, where each element is the {method}
        of all the previous values in this row or column.
    """,
    refer_to_module_name="DataFrame",
)

doc_resample = partial(
    doc_qc_method,
    template="""
    Resample time-series data and apply aggregation on it.

    Group data into intervals by time-series row/column with
    a specified frequency and {action}.

    Parameters
    ----------
    resample_kwargs : dict
        Resample parameters as expected by ``modin.pandas.DataFrame.resample`` signature.
    {extra_params}
    Returns
    -------
    BaseQueryCompiler
        New QueryCompiler containing the result of resample aggregation built by the
        following rules:

        {build_rules}
    """,
    refer_to_module_name="resample.Resampler",
)


def doc_resample_reduce(result, refer_to, params=None, compatibility_params=True):
    """
    Build decorator which adds docstring for the resample reduce method.

    Parameters
    ----------
    result : str
        The result of the method.
    refer_to : str
        Method name in ``modin.pandas.resample.Resampler`` module to refer to for
        more information about parameters and output format.
    params : str, optional
        Method parameters in the NumPy docstyle format to substitute
        to the docstring template.
    compatibility_params : bool, default: True
        Whether method takes `*args` and `**kwargs` that do not affect
        the result.

    Returns
    -------
    callable
    """
    action = f"compute {result} for each group"

    params_substitution = (
        (
            """
        *args : iterable
            Serves the compatibility purpose. Does not affect the result.
        **kwargs : dict
            Serves the compatibility purpose. Does not affect the result.
        """
        )
        if compatibility_params
        else ""
    )

    if params:
        params_substitution = format_string(
            "{params}\n{params_substitution}",
            params=params,
            params_substitution=params_substitution,
        )

    build_rules = f"""
            - Labels on the specified axis are the group names (time-stamps)
            - Labels on the opposite of specified axis are preserved.
            - Each element of QueryCompiler is the {result} for the
              corresponding group and column/row."""
    return doc_resample(
        action=action,
        extra_params=params_substitution,
        build_rules=build_rules,
        refer_to=refer_to,
    )


def doc_resample_agg(action, output, refer_to, params=None):
    """
    Build decorator which adds docstring for the resample aggregation method.

    Parameters
    ----------
    action : str
        What method does with the resampled data.
    output : str
        What is the content of column names in the result.
    refer_to : str
        Method name in ``modin.pandas.resample.Resampler`` module to refer to for
        more information about parameters and output format.
    params : str, optional
        Method parameters in the NumPy docstyle format to substitute
        to the docstring template.

    Returns
    -------
    callable
    """
    action = f"{action} for each group over the specified axis"

    params_substitution = """
        *args : iterable
            Positional arguments to pass to the aggregation function.
        **kwargs : dict
            Keyword arguments to pass to the aggregation function.
        """

    if params:
        params_substitution = format_string(
            "{params}\n{params_substitution}",
            params=params,
            params_substitution=params_substitution,
        )

    build_rules = f"""
            - Labels on the specified axis are the group names (time-stamps)
            - Labels on the opposite of specified axis are a MultiIndex, where first level
              contains preserved labels of this axis and the second level is the {output}.
            - Each element of QueryCompiler is the result of corresponding function for the
              corresponding group and column/row."""
    return doc_resample(
        action=action,
        extra_params=params_substitution,
        build_rules=build_rules,
        refer_to=refer_to,
    )


def doc_resample_fillna(method, refer_to, params=None, overwrite_template_params=False):
    """
    Build decorator which adds docstring for the resample fillna query compiler method.

    Parameters
    ----------
    method : str
        Fillna method name.
    refer_to : str
        Method name in ``modin.pandas.resample.Resampler`` module to refer to for
        more information about parameters and output format.
    params : str, optional
        Method parameters in the NumPy docstyle format to substitute
        to the docstring template.
    overwrite_template_params : bool, default: False
        If `params` is specified indicates whether to overwrite method parameters in
        the docstring template or append then at the end.

    Returns
    -------
    callable
    """
    action = f"fill missing values in each group independently using {method} method"
    params_substitution = "limit : int\n"

    if params:
        params_substitution = (
            params
            if overwrite_template_params
            else format_string(
                "{params}\n{params_substitution}",
                params=params,
                params_substitution=params_substitution,
            )
        )

    build_rules = "- QueryCompiler contains unsampled data with missing values filled."

    return doc_resample(
        action=action,
        extra_params=params_substitution,
        build_rules=build_rules,
        refer_to=refer_to,
    )


doc_dt = partial(
    doc_qc_method,
    template="""
    Get {prop} for each {dt_type} value.
    {params}
    Returns
    -------
    BaseQueryCompiler
        New QueryCompiler with the same shape as `self`, where each element is
        {prop} for the corresponding {dt_type} value.
    """,
    one_column_method=True,
    refer_to_module_name="Series.dt",
)

doc_dt_timestamp = partial(doc_dt, dt_type="datetime")
doc_dt_interval = partial(doc_dt, dt_type="interval")
doc_dt_period = partial(doc_dt, dt_type="period")

doc_dt_round = partial(
    doc_qc_method,
    template="""
    Perform {refer_to} operation on the underlying time-series data to the specified `freq`.

    Parameters
    ----------
    freq : str
    ambiguous : {{"raise", "infer", "NaT"}} or bool mask, default: "raise"
    nonexistent : {{"raise", "shift_forward", "shift_backward", "NaT"}} or timedelta, default: "raise"

    Returns
    -------
    BaseQueryCompiler
        New QueryCompiler with performed {refer_to} operation on every element.
    """,
    one_column_method=True,
    refer_to_module_name="Series.dt",
)

doc_str_method = partial(
    doc_qc_method,
    template="""
    Apply "{refer_to}" function to each string value in QueryCompiler.
    {params}
    Returns
    -------
    BaseQueryCompiler
        New QueryCompiler containing the result of execution of the "{refer_to}" function
        against each string element.
    """,
    one_column_method=True,
    refer_to_module_name="Series.str",
)


def doc_window_method(
    window_cls_name,
    result,
    refer_to,
    action=None,
    win_type="rolling window",
    params=None,
    build_rules="aggregation",
):
    """
    Build decorator which adds docstring for a window method.

    Parameters
    ----------
    window_cls_name : str
        The Window class the method is on.
    result : str
        The result of the method.
    refer_to : str
        Method name in ``modin.pandas.window.Window`` module to refer to
        for more information about parameters and output format.
    action : str, optional
        What method does with the created window.
    win_type : str, default: "rolling_window"
        Type of window that the method creates.
    params : str, optional
        Method parameters in the NumPy docstyle format to substitute
        to the docstring template.
    build_rules : str, default: "aggregation"
        Description of the data output format.

    Returns
    -------
    callable
    """
    template = """
        Create {win_type} and {action} for each window over the given axis.

        Parameters
        ----------
        fold_axis : {{0, 1}}
        {window_args_name} : list
            Rolling windows arguments with the same signature as ``modin.pandas.DataFrame.rolling``.
        {extra_params}
        Returns
        -------
        BaseQueryCompiler
            New QueryCompiler containing {result} for each window, built by the following
            rules:

            {build_rules}
        """
    doc_build_rules = {
        "aggregation": f"""
            - Output QueryCompiler has the same shape and axes labels as the source.
            - Each element is the {result} for the corresponding window.""",
        "udf_aggregation": """
            - Labels on the specified axis are preserved.
            - Labels on the opposite of specified axis are MultiIndex, where first level
              contains preserved labels of this axis and the second level has the function names.
            - Each element of QueryCompiler is the result of corresponding function for the
              corresponding window and column/row.""",
    }
    if action is None:
        action = f"compute {result}"
    if win_type == "rolling window":
        window_args_name = "rolling_kwargs"
    elif win_type == "expanding window":
        window_args_name = "expanding_args"
    else:
        window_args_name = "window_kwargs"

    # We need that `params` value ended with new line to have
    # an empty line between "parameters" and "return" sections
    if params and params[-1] != "\n":
        params += "\n"

    if params is None:
        params = ""

    return doc_qc_method(
        template,
        result=result,
        action=action,
        win_type=win_type,
        extra_params=params,
        build_rules=doc_build_rules.get(build_rules, build_rules),
        refer_to=f"{window_cls_name}.{refer_to}",
        window_args_name=window_args_name,
    )


def doc_groupby_method(result, refer_to, action=None):
    """
    Build decorator which adds docstring for the groupby reduce method.

    Parameters
    ----------
    result : str
        The result of reduce.
    refer_to : str
        Method name in ``modin.pandas.groupby`` module to refer to
        for more information about parameters and output format.
    action : str, optional
        What method does with groups.

    Returns
    -------
    callable
    """
    template = """
    Group QueryCompiler data and {action} for every group.

    Parameters
    ----------
    by : BaseQueryCompiler, column or index label, Grouper or list of such
        Object that determine groups.
    axis : {{0, 1}}
        Axis to group and apply aggregation function along.
        0 is for index, when 1 is for columns.
    groupby_kwargs : dict
        GroupBy parameters as expected by ``modin.pandas.DataFrame.groupby`` signature.
    agg_args : list-like
        Positional arguments to pass to the `agg_func`.
    agg_kwargs : dict
        Key arguments to pass to the `agg_func`.
    drop : bool, default: False
        If `by` is a QueryCompiler indicates whether or not by-data came
        from the `self`.

    Returns
    -------
    BaseQueryCompiler
        QueryCompiler containing the result of groupby reduce built by the
        following rules:

        - Labels on the opposite of specified axis are preserved.
        - If groupby_args["as_index"] is True then labels on the specified axis
          are the group names, otherwise labels would be default: 0, 1 ... n.
        - If groupby_args["as_index"] is False, then first N columns/rows of the frame
          contain group names, where N is the columns/rows to group on.
        - Each element of QueryCompiler is the {result} for the
          corresponding group and column/row.

    .. warning
        `map_args` and `reduce_args` parameters are deprecated. They're leaked here from
        ``PandasQueryCompiler.groupby_*``, pandas storage format implements groupby via TreeReduce
        approach, but for other storage formats these parameters make no sense, and so they'll be removed in the future.
    """
    if action is None:
        action = f"compute {result}"

    return doc_qc_method(
        template, result=result, action=action, refer_to=f"GroupBy.{refer_to}"
    )
