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

from sklearn.utils.fixes import threadpool_info
from sklearn.utils._show_versions import _get_sys_info
from sklearn.utils._show_versions import _get_deps_info
from sklearn.utils._show_versions import show_versions
from sklearn.utils._testing import ignore_warnings


def test_get_sys_info():
    sys_info = _get_sys_info()

    assert "python" in sys_info
    assert "executable" in sys_info
    assert "machine" in sys_info


def test_get_deps_info():
    with ignore_warnings():
        deps_info = _get_deps_info()

    assert "pip" in deps_info
    assert "setuptools" in deps_info
    assert "sklearn" in deps_info
    assert "numpy" in deps_info
    assert "scipy" in deps_info
    assert "Cython" in deps_info
    assert "pandas" in deps_info
    assert "matplotlib" in deps_info
    assert "joblib" in deps_info
    # assert "modin" in deps_info


def test_show_versions(capsys):
    with ignore_warnings():
        show_versions()
        out, err = capsys.readouterr()

    assert "python" in out
    assert "numpy" in out

    info = threadpool_info()
    if info:
        assert "threadpoolctl info:" in out
