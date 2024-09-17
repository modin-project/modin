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

from modin.config import Engine
from modin.config.pubsub import context
from modin.tests.pandas.utils import (
    NoModinException,
    create_test_dfs,
    create_test_series,
    df_equals,
)
from modin.utils import try_cast_to_pandas


def create_test_df_in_defined_mode(
    *args, post_fn=None, backend=None, df_mode=None, **kwargs
):
    with context(NativeDataframeMode=df_mode):
        return create_test_dfs(*args, post_fn=post_fn, backend=backend, **kwargs)


def create_test_series_in_defined_mode(
    vals, sort=False, backend=None, df_mode=None, **kwargs
):
    with context(NativeDataframeMode=df_mode):
        return create_test_series(vals, sort=sort, backend=backend, **kwargs)


def eval_general_interop(
    data,
    backend,
    operation,
    df_mode_pair,
    comparator=df_equals,
    __inplace__=False,
    expected_exception=None,
    check_kwargs_callable=True,
    md_extra_kwargs=None,
    comparator_kwargs=None,
    **kwargs,
):
    df_mode1, df_mode2 = df_mode_pair
    modin_df1, pandas_df1 = create_test_df_in_defined_mode(
        data, backend=backend, df_mode=df_mode1
    )
    modin_df2, pandas_df2 = create_test_df_in_defined_mode(
        data, backend=backend, df_mode=df_mode2
    )
    md_kwargs, pd_kwargs = {}, {}

    def execute_callable(fn, inplace=False, md_kwargs={}, pd_kwargs={}):
        try:
            pd_result = fn(pandas_df1, pandas_df2, **pd_kwargs)
        except Exception as pd_e:
            try:
                if inplace:
                    _ = fn(modin_df1, modin_df2, **md_kwargs)
                    try_cast_to_pandas(modin_df1)  # force materialization
                else:
                    try_cast_to_pandas(
                        fn(modin_df1, modin_df2, **md_kwargs)
                    )  # force materialization
            except Exception as md_e:
                assert isinstance(
                    md_e, type(pd_e)
                ), "Got Modin Exception type {}, but pandas Exception type {} was expected".format(
                    type(md_e), type(pd_e)
                )
                if expected_exception:
                    if Engine.get() == "Ray":
                        from ray.exceptions import RayTaskError

                        # unwrap ray exceptions from remote worker
                        if isinstance(md_e, RayTaskError):
                            md_e = md_e.args[0]
                    assert (
                        type(md_e) is type(expected_exception)
                        and md_e.args == expected_exception.args
                    ), f"not acceptable Modin's exception: [{repr(md_e)}]"
                    assert (
                        pd_e.args == expected_exception.args
                    ), f"not acceptable Pandas' exception: [{repr(pd_e)}]"
                elif expected_exception is False:
                    # The only way to disable exception message checking.
                    pass
                else:
                    # It’s not enough that Modin and pandas have the same types of exceptions;
                    # we need to explicitly specify the instance of an exception
                    # (using `expected_exception`) in tests so that we can check exception messages.
                    # This allows us to eliminate situations where exceptions are thrown
                    # that we don't expect, which could hide different bugs.
                    raise pd_e
            else:
                raise NoModinException(
                    f"Modin doesn't throw an exception, while pandas does: [{repr(pd_e)}]"
                )
        else:
            md_result = fn(modin_df1, modin_df2, **md_kwargs)
            return (md_result, pd_result) if not inplace else (modin_df1, pandas_df1)

    for key, value in kwargs.items():
        if check_kwargs_callable and callable(value):
            values = execute_callable(value)
            # that means, that callable raised an exception
            if values is None:
                return
            else:
                md_value, pd_value = values
        else:
            md_value, pd_value = value, value

        md_kwargs[key] = md_value
        pd_kwargs[key] = pd_value

        if md_extra_kwargs:
            assert isinstance(md_extra_kwargs, dict)
            md_kwargs.update(md_extra_kwargs)

    values = execute_callable(
        operation, md_kwargs=md_kwargs, pd_kwargs=pd_kwargs, inplace=__inplace__
    )
    if values is not None:
        comparator(*values, **(comparator_kwargs or {}))
