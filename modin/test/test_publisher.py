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

from collections import defaultdict

from modin import Publisher


def test_equals():
    pub = Publisher("name", "value1")
    assert pub.get() == "Value1"

    pub.put("value2")
    assert pub.get() == "Value2"


def test_triggers():
    results = defaultdict(int)
    callbacks = []

    def make_callback(name, res=results):
        def callback(p: Publisher):
            res[name] += 1

        # keep reference to callbacks so they won't be removed by GC
        callbacks.append(callback)
        return callback

    pub = Publisher("name", "init")
    pub.once("init", make_callback("init"))
    assert results["init"] == 1

    pub.once("never", make_callback("never"))
    pub.once("once", make_callback("once"))
    pub.subscribe(make_callback("subscribe"))

    pub.put("multi")
    pub.put("once")
    pub.put("multi")
    pub.put("once")

    expected = [("init", 1), ("never", 0), ("once", 1), ("subscribe", 5)]
    for name, val in expected:
        assert results[name] == val, "{} has wrong count".format(name)
