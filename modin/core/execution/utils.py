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

"""General utils for execution module."""

import contextlib
import os

from modin.error_message import ErrorMessage


@contextlib.contextmanager
def set_env(**environ):
    """
    Temporarily set the process environment variables.
    """
    old_environ = os.environ.copy()
    os.environ.update(environ)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(old_environ)


if "_MODIN_DOC_CHECKER_" in os.environ:

    # The doc checker should get the non-processed functions
    def remote_function(func, ignore_defaults=False):
        return func


# Check if the function already exists to avoid circular imports
elif "remote_function" not in dir():
    # TODO(https://github.com/modin-project/modin/issues/7429): Use
    # frame-level engine config.

    from modin.config import Engine

    if Engine.get() == "Ray":
        from modin.core.execution.ray.common import RayWrapper

        _preprocess_func = RayWrapper.put
    elif Engine.get() == "Unidist":
        from modin.core.execution.unidist.common import UnidistWrapper

        _preprocess_func = UnidistWrapper.put
    elif Engine.get() == "Dask":
        from modin.core.execution.dask.common import DaskWrapper

        # The function cache is not supported for Dask
        def remote_function(func, ignore_defaults=False):
            return DaskWrapper.put(func)

    else:

        def remote_function(func, ignore_defaults=False):
            return func

    if "remote_function" not in dir():
        _remote_function_cache = {}

        def remote_function(func, ignore_defaults=False):  # noqa: F811
            if "<locals>" in func.__qualname__:  # Nested function
                if func.__closure__:
                    ErrorMessage.single_warning(
                        f"The nested function {func} can not be cached, because "
                        + "it captures objects from the outer scope."
                    )
                    return func
                if not ignore_defaults and func.__defaults__:
                    ErrorMessage.single_warning(
                        f"The nested function {func} can not be cached, because it has "
                        + "default values. Use `ignore_defaults` to forcibly enable caching."
                    )
                    return func
                # For the nested functions, use __code__ as the key, because it's the same
                # object for each instance of the function.
                key = id(func.__code__)
            else:
                key = func
            ref = _remote_function_cache.get(key, None)
            if ref is None:
                ref = _preprocess_func(func)
                _remote_function_cache[key] = ref
            return ref
