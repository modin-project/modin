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

from urllib.request import urlopen
from urllib.error import HTTPError
from concurrent.futures import ThreadPoolExecutor

from modin.config import DocstringUrlTestMode
import modin.utils

assert DocstringUrlTestMode.get(), "Docstring URL test mode has to be enabled"

from modin.pandas import *  # ensure all docstring are generated


def test_all_urls_exist():

    broken = []

    def _test_url(url, broken):
        try:
            with urlopen(url):
                pass
        except HTTPError:
            broken.append(url)

    with ThreadPoolExecutor(32) as pool:
        pool.map(_test_url, modin.utils._GENERATED_URLS)

    assert not broken, "Invalid URLs detected"
