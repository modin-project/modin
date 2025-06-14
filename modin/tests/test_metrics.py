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

from typing import Union

import pytest

import modin.logging
import modin.pandas as pd
from modin.config import MetricsMode
from modin.logging.metrics import (
    _metric_handlers,
    add_metric_handler,
    clear_metric_handler,
    emit_metric,
)


class FakeTelemetryClient:

    def __init__(self):
        self._metrics = {}
        self._metric_handler = None

    def metric_handler_fail(self, name: str, value: Union[int, float]):
        raise KeyError("Poorly implemented metric handler")

    def metric_handler_pass(self, name: str, value: Union[int, float]):
        self._metrics[name] = value


@modin.logging.enable_logging
def func(do_raise):
    if do_raise:
        raise ValueError()


@pytest.fixture()
def metric_client():
    MetricsMode.enable()
    client = FakeTelemetryClient()
    yield client
    clear_metric_handler(client._metric_handler)
    MetricsMode.disable()


def test_metrics_api_timings(metric_client):
    assert len(_metric_handlers) == 0
    metric_client._metric_handler = metric_client.metric_handler_pass
    add_metric_handler(metric_client._metric_handler)
    assert len(_metric_handlers) == 1
    assert _metric_handlers[0] == metric_client._metric_handler
    func(do_raise=False)
    assert len(metric_client._metrics) == 1
    assert metric_client._metrics["modin.pandas-api.func"] is not None
    assert metric_client._metrics["modin.pandas-api.func"] > 0.0


def test_df_metrics(metric_client):
    metric_client._metric_handler = metric_client.metric_handler_pass
    add_metric_handler(metric_client._metric_handler)
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    df.sum()
    assert len(metric_client._metrics) == 55
    assert metric_client._metrics["modin.pandas-api.dataframe.sum"] is not None
    assert metric_client._metrics["modin.pandas-api.dataframe.sum"] > 0.0


def test_metrics_handler_fails(metric_client):
    assert len(metric_client._metrics) == 0
    metric_client._metric_handler = metric_client.metric_handler_fail
    add_metric_handler(metric_client._metric_handler)
    assert len(_metric_handlers) == 1
    func(do_raise=False)
    assert len(_metric_handlers) == 0
    assert len(metric_client._metrics) == 0


def test_emit_name_enforced():
    MetricsMode.enable()
    with pytest.raises(KeyError):
        emit_metric("Not::A::Valid::Metric::Name", 1.0)


def test_metrics_can_be_opt_out(metric_client):
    MetricsMode.enable()
    assert len(metric_client._metrics) == 0
    metric_client._metric_handler = metric_client.metric_handler_pass
    add_metric_handler(metric_client._metric_handler)
    # If Metrics are disabled after the addition of a handler
    # no metrics are emitted
    MetricsMode.disable()
    assert len(_metric_handlers) == 1
    func(do_raise=False)
    assert len(metric_client._metrics) == 0
