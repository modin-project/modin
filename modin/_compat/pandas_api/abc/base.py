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

"""Module for interface for class compatibility layer for Dataset (common DataFrame/Series)."""

from modin.logging import ClassLogger


class BaseCompatibleBasePandasDataset(ClassLogger):
    """Interface for compatibility layer for Dataset."""

    def max(self, *args, **kwargs):  # noqa: GL08
        pass

    def min(self, *args, **kwargs):  # noqa: GL08
        pass

    def mean(self, *args, **kwargs):  # noqa: GL08
        pass

    def median(self, *args, **kwargs):  # noqa: GL08
        pass

    def rank(self, *args, **kwargs):  # noqa: GL08
        pass

    def reindex(self, *args, **kwargs):  # noqa: GL08
        pass

    def rolling(self, *args, **kwargs):  # noqa: GL08
        pass

    def sample(self, *args, **kwargs):  # noqa: GL08
        pass

    def sem(self, *args, **kwargs):  # noqa: GL08
        pass

    def shift(self, *args, **kwargs):  # noqa: GL08
        pass

    def skew(self, *args, **kwargs):  # noqa: GL08
        pass

    def std(self, *args, **kwargs):  # noqa: GL08
        pass

    def to_csv(self, *args, **kwargs):  # noqa: GL08
        pass

    def to_json(self, *args, **kwargs):  # noqa: GL08
        pass

    def to_markdown(self, *args, **kwargs):  # noqa: GL08
        pass

    def to_latex(self, *args, **kwargs):  # noqa: GL08
        pass

    def to_pickle(self, *args, **kwargs):  # noqa: GL08
        pass

    def var(self, *args, **kwargs):  # noqa: GL08
        pass
