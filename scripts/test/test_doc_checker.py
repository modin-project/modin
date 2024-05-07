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

import pytest

from scripts.doc_checker import (
    MODIN_ERROR_CODES,
    check_optional_args,
    check_spelling_words,
    construct_validator,
    get_noqa_checks,
    get_optional_args,
)


@pytest.mark.parametrize(
    "import_path, result",
    [
        ("scripts.test.examples.optional_square", {"number": 5}),
        ("scripts.test.examples.optional_square_empty_parameters", {"number": 5}),
        ("scripts.test.examples.square_summary", {}),
        ("scripts.test.examples.weakdict", {}),
        ("scripts.test.examples", {}),
    ],
)
def test_get_optional_args(import_path, result):
    optional_args = get_optional_args(construct_validator(import_path))
    assert optional_args == result


@pytest.mark.parametrize(
    "import_path, result",
    [
        (
            "scripts.test.examples.optional_square",
            [
                (
                    "MD01",
                    MODIN_ERROR_CODES["MD01"].format(parameter="number", found="int"),
                )
            ],
        ),
        ("scripts.test.examples.optional_square_empty_parameters", []),
        ("scripts.test.examples.square_summary", []),
        ("scripts.test.examples.weakdict", []),
        ("scripts.test.examples", []),
    ],
)
def test_check_optional_args(import_path, result):
    errors = check_optional_args(construct_validator(import_path))
    assert errors == result


@pytest.mark.parametrize(
    "import_path, result",
    [
        ("scripts.test.examples.optional_square", []),
        (
            "scripts.test.examples.square_summary",
            [
                ("MD02", 57, "Pandas", "pandas"),
                ("MD02", 57, "Numpy", "NumPy"),
            ],
        ),
        ("scripts.test.examples.optional_square_empty_parameters", []),
        ("scripts.test.examples.weakdict", []),
        ("scripts.test.examples", []),
    ],
)
def test_check_spelling_words(import_path, result):
    result_errors = []
    for code, line, word, reference in result:
        result_errors.append(
            (
                code,
                MODIN_ERROR_CODES[code].format(
                    line=line, word=word, reference=reference
                ),
            )
        )
    errors = check_spelling_words(construct_validator(import_path))
    # the order of incorrect words found on the same line is not guaranteed
    for error in errors:
        assert error in result_errors


@pytest.mark.parametrize(
    "import_path, result",
    [
        ("scripts.test.examples.optional_square", ["all"]),
        ("scripts.test.examples.optional_square_empty_parameters", []),
        ("scripts.test.examples.square_summary", ["PR01", "GL08"]),
        ("scripts.test.examples.weakdict", ["GL08"]),
        ("scripts.test.examples", ["MD02"]),
    ],
)
def test_get_noqa_checks(import_path, result):
    noqa_checks = get_noqa_checks(construct_validator(import_path))
    assert noqa_checks == result
