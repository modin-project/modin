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

"""Module houses class responsible for emulating execution of remote operations for python execution."""


class PythonWrapper:
    """The class responsible for execution of remote operations."""

    @classmethod
    def deploy(cls, func, f_args=None, f_kwargs=None, num_returns=1):
        args = [] if f_args is None else f_args
        kwargs = {} if f_kwargs is None else f_kwargs
        return func(*args, **kwargs)

    @classmethod
    def materialize(cls, data):
        return data.copy() if hasattr(data, "copy") else data

    @classmethod
    def put(cls, data, **kwargs):
        return data
