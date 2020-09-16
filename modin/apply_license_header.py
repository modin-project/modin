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

import os
from os.path import dirname, abspath


# This is the python file root directory (modin/modin)
rootdir = dirname(abspath(__file__))
exclude_files = ["_version.py"]

with open("{}{}".format(dirname(rootdir), "/LICENSE_HEADER"), "r") as f:
    # Lines to check each line individually
    header_lines = f.readlines()

for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        filepath = os.path.join(subdir, file)
        if file.endswith(".py") and file not in exclude_files:
            with open(filepath, "r+") as f:
                # Lines for line by line comparison
                py_file_lines = f.readlines()
                if len(py_file_lines):
                    first_non_comment_idx = next(
                        i
                        for i in range(len(py_file_lines))
                        if not py_file_lines[i].startswith("# ")
                    )
                    py_file_lines = py_file_lines[first_non_comment_idx:]
                new_contents = "".join(header_lines + py_file_lines)
                f.seek(0)
                f.write(new_contents)
                f.truncate()
