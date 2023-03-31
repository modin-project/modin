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
import pandas
import pytz
import timeit
from random import randint, uniform, choice

from ..dataframe.utils import ColNameCodec

UNICODE_RANGES = [
    (0x0020, 0x007F),  # Basic Latin
    (0x00A0, 0x00FF),  # Latin-1 Supplement
    (0x0100, 0x017F),  # Latin Extended-A
    (0x0180, 0x024F),  # Latin Extended-B
    (0x0250, 0x02AF),  # IPA Extensions
    (0x02B0, 0x02FF),  # Spacing Modifier Letters
    (0x0300, 0x036F),  # Combining Diacritical Marks
    (0x0370, 0x03FF),  # Greek and Coptic
    (0x10330, 0x1034F),  # Gothic
    (0xE0000, 0xE007F),  # Tags
]
UNICODE_ALPHABET = [chr(c) for r in UNICODE_RANGES for c in range(r[0], r[1] + 1)]


def test_encode_col_name():
    def test(name):
        encoded = ColNameCodec.encode(name)
        assert ColNameCodec.decode(encoded) == name

    test("")
    test(None)
    test(("", ""))

    for i in range(0, 1000):
        test(randint(-sys.maxsize, sys.maxsize))
    for i in range(0, 1000):
        test(uniform(-sys.maxsize, sys.maxsize))
    for i in range(0, 1000):
        test(rnd_unicode(randint(0, 100)))
    for i in range(0, 1000):
        test((rnd_unicode(randint(0, 100)), rnd_unicode(randint(0, 100))))
    for i in range(0, 1000):
        tz = choice(pytz.all_timezones)
        test(pandas.Timestamp(randint(0, 0xFFFFFFFF), unit="s", tz=tz))


def rnd_unicode(length):
    return "".join(choice(UNICODE_ALPHABET) for _ in range(length))


def test_time():
    ranges = [
        (0x0041, 0x005A),  # Alpha chars
        (0x0020, 0x007F),  # Basic Latin
        (0x00A0, 0x00FF),  # Latin-1 Supplement
    ]
    repeat = 10
    text_len = 100000

    for r in ranges:
        alphabet = "".join([chr(c) for c in range(r[0], r[1] + 1)])
        text = (
            alphabet * int(text_len / len(alphabet))
            + alphabet[0 : divmod(text_len, len(alphabet))[1]]
        )
        encoded_text = ColNameCodec.encode(text)
        assert text == ColNameCodec.decode(encoded_text)
        print(f"Alphabet: {alphabet}")  # noqa: T201
        print(f"Text len: {len(text)}")  # noqa: T201
        print(f"Encoded text len: {len(encoded_text)}")  # noqa: T201

        def test_encode():
            ColNameCodec.encode(text)

        def test_decode():
            ColNameCodec.decode(encoded_text)

        time = timeit.timeit(stmt=test_encode, number=repeat)
        print(f"Encode time: {time/repeat} seconds")  # noqa: T201
        time = timeit.timeit(stmt=test_decode, number=repeat)
        print(f"Decode time: {time/repeat} seconds")  # noqa: T201
        print("--------------------------------------")  # noqa: T201
