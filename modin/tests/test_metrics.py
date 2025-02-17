import pytest

import modin.logging
from modin.config import TelemetryMode
from modin.logging.metrics import add_metric_handler, add_telemetry_handler

class FakeTelemetryClient:
    
    _metrics = {}
    def metric_handler(self, name:str, value:int|float):
        self._metrics[name] = value

@modin.logging.enable_logging
def func(do_raise):
    if do_raise:
        raise ValueError()

def test_metrics_api_timings():
    client = FakeTelemetryClient
    add_metric_handler(client.metric_handler)
    func()
    assert len(client._metrics) == 1
    pass

def test_metrics_handler_fails():
    pass

def test_metrics_handler_timeout():
    pass

def test_metrics_can_be_opt_out():
    pass

def test_metrics_emit_when_logging_disabled():
    pass