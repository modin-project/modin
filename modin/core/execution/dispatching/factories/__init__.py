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

"""Factories responsible for dispatching to specific execution."""

from . import factories


def _get_remote_engines():
    """Yield engines of all of the experimental remote factories."""
    for name in dir(factories):
        obj = getattr(factories, name)
        if isinstance(obj, type) and issubclass(
            obj, factories.ExperimentalRemoteFactory
        ):
            try:
                yield obj.get_info().engine
            except factories.NotRealFactory:
                pass


REMOTE_ENGINES = set(_get_remote_engines())
