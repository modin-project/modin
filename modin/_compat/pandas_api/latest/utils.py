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

"""Module for utility functions used by 'latest pandas' compatibility layer."""

from pandas._libs.lib import no_default, NoDefault


def create_stat_method(name):
    """
    Create statistical method from operation name.

    Parameters
    ----------
    name : str
        Operation name to perform.

    Returns
    -------
    callable
        Method to perform given statistics, for more see ``BasePandasDataset._stat_operation``.
    """

    def stat_method(
        self,
        axis: "int | None | NoDefault" = no_default,
        skipna=True,
        level=None,
        numeric_only=None,
        **kwargs,
    ):
        return self._stat_operation(name, axis, skipna, level, numeric_only, **kwargs)

    return stat_method
