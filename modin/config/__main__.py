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

from . import *  # noqa: F403, F401
from .pubsub import Parameter


def print_config_help():
    for objname in sorted(globals()):
        obj = globals()[objname]
        if isinstance(obj, type) and issubclass(obj, Parameter) and not obj.is_abstract:
            print(f"{obj.get_help()}\n\tCurrent value: {obj.get()}")  # noqa: T001


if __name__ == "__main__":
    print_config_help()
