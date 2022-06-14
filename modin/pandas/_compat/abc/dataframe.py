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

from abc import ABC

from modin.logging import LoggerBase


class BaseCompatibilityDataFrame(ABC, LoggerBase):
    def applymap(self, *args, **kwargs):
        pass

    def apply(self, *args, **kwargs):
        pass

    def info(self, *args, **kwargs):
        pass

    def pivot_table(self, *args, **kwargs):
        pass

    def prod(self, *args, **kwargs):
        pass

    def replace(self, *args, **kwargs):
        pass

    def sum(self, *args, **kwargs):
        pass

    def to_parquet(self, *args, **kwargs):
        pass
