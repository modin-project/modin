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

"""Module for 'latest pandas' compatibility layer for window objects."""

import pandas.core.window.rolling

from ..abc.window import BaseCompatibleWindow, BaseCompatibleRolling
from modin.utils import _inherit_docstrings, append_to_docstring


@append_to_docstring("Compatibility layer for 'latest pandas' for Window.")
@_inherit_docstrings(pandas.core.window.rolling.Window)
class LatestCompatibleWindow(BaseCompatibleWindow):
    def __init__(
        self,
        dataframe,
        window=None,
        min_periods=None,
        center=False,
        win_type=None,
        on=None,
        axis=0,
        closed=None,
        step=None,
        method="single",
        **kwargs,
    ):
        self._init(
            dataframe,
            [
                (window, min_periods, center, win_type, on, axis, closed, step, method),
                kwargs,
            ],
            axis,
        )


@append_to_docstring("Compatibility layer for 'latest pandas' for Rolling.")
@_inherit_docstrings(pandas.core.window.rolling.Rolling)
class LatestCompatibleRolling(BaseCompatibleRolling):
    def __init__(
        self,
        dataframe,
        window=None,
        min_periods=None,
        center=False,
        win_type=None,
        on=None,
        axis=0,
        closed=None,
        step=None,
        method="single",
        **kwargs,
    ):
        self._init(
            dataframe,
            [
                (window, min_periods, center, win_type, on, axis, closed, step, method),
                kwargs,
            ],
            axis,
        )
