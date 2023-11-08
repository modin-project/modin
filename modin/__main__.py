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

"""Command-line interface piece, called when user issues "python -m modin --foo"."""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(
        "python -m modin",
        description="Drop-in pandas replacement; refer to https://modin.readthedocs.io/ for details.",
    )
    parser.add_argument(
        "--versions",
        action="store_true",
        default=False,
        help="Show versions of all known components",
    )

    args = parser.parse_args()
    if args.versions:
        from modin.utils import show_versions

        show_versions()


if __name__ == "__main__":
    main()
