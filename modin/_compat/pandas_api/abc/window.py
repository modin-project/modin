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

"""Module for interface for class compatibility layer for window objects."""

from modin.logging.class_logger import ClassLogger


class BaseCompatibleWindow(ClassLogger):
    """Interface for class compatibility layer for Window."""

    def _init(self, dataframe, window_args, axis):
        """
        Initialize the Window object for real.

        Utilize translated potentially pandas-specific arguments.

        Parameters
        ----------
        dataframe : DataFrame
            The dataframe object to apply window functions against.
        window_args : sequence
            The arguments to be passed to .Window() except dataframe.
        axis : {0, 1}
            The axis to build Window against.
        """
        pass


class BaseCompatibleRolling(ClassLogger):
    """Interface for class compatibility layer for Rolling."""

    def _init(self, dataframe, rolling_args, axis):
        """
        Initialize the Rolling object for real.

        Utilize translated potentially pandas-specific arguments.

        Parameters
        ----------
        dataframe : DataFrame
            The dataframe object to apply rolling functions against.
        rolling_args : sequence
            The arguments to be passed to .Rolling() except dataframe.
        axis : {0, 1}
            The axis to build Rolling against.
        """
        pass
