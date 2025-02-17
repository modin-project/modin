from time import sleep
import pytest

import modin.logging
from modin.config import MetricsMode
from modin.logging.metrics import (
    add_metric_handler,
    _metric_handlers,
    clear_metric_handler,
)


class FakeTelemetryClient:

    def __init__(self):
        self._metrics = {}
        self._metric_handler = None

    def metric_handler_fail(self, name: str, value: int | float):
        raise KeyError("Poorly implemented metric handler")

    def metric_handler_timeout(self, name: str, value: int | float):
        sleep(0.500)
        self.metric_handler_pass(name, value)

    def metric_handler_pass(self, name: str, value: int | float):
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


def test_metrics_handler_fails(metric_client):
    assert len(metric_client._metrics) == 0
    metric_client._metric_handler = metric_client.metric_handler_fail
    add_metric_handler(metric_client._metric_handler)
    assert len(_metric_handlers) == 1
    func(do_raise=False)
    assert len(_metric_handlers) == 0
    assert len(metric_client._metrics) == 0


def test_metrics_handler_timeout(metric_client):
    assert len(metric_client._metrics) == 0
    metric_client._metric_handler = metric_client.metric_handler_timeout
    add_metric_handler(metric_client._metric_handler)
    assert len(_metric_handlers) == 1
    func(do_raise=False)
    assert len(_metric_handlers) == 0
    assert len(metric_client._metrics) == 0


def test_metrics_can_be_opt_out(metric_client):
    # Cannot register a metrics handler if metrics are
    # disabled
    MetricsMode.disable()
    assert len(metric_client._metrics) == 0
    metric_client._metric_handler = metric_client.metric_handler_pass
    add_metric_handler(metric_client._metric_handler)
    assert len(_metric_handlers) == 0
    # If Metrics are disabled after the addition of a handler
    # no metrics are emitted
    MetricsMode.enable()
    add_metric_handler(metric_client._metric_handler)
    MetricsMode.disable()
    assert len(_metric_handlers) == 1
    func(do_raise=False)
    assert len(metric_client._metrics) == 0
