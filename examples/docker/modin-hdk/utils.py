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

import sys
import time
from os.path import abspath, dirname, join

MODIN_DIR = abspath(join(dirname(__file__), *[".." for _ in range(3)]))
if MODIN_DIR not in sys.path:
    sys.path.insert(0, MODIN_DIR)


def measure(name, func, *args, **kwargs):
    t0 = time.time()
    res = func(*args, **kwargs)
    t1 = time.time()
    print(f"{name}: {t1 - t0} sec")
    return res
