import re
from typing import Callable, Union

from modin.config.envvars import MetricsMode
from modin.utils import timeout


metric_name_pattern = r"[a-zA-Z\._\-0-9]+$"
_metric_handlers: list[Callable[[str, Union[int, float]], None]] = []


# Metric/Telemetry hooks can be implemented by plugin engines
# to collect discrete data on how modin is performing at the
# high level modin layer.
def emit_metric(name: str, value: Union[int, float]) -> None:
    """
    emit a metric using the set of registered handlers
    """
    if MetricsMode.get() == "disable":
        return
    if not re.fullmatch(metric_name_pattern, name):
        raise KeyError(
            f"Metrics name is not in metric-name dot format, (eg. modin.dataframe.hist.duration ): {name}"
        )

    handlers = _metric_handlers.copy()
    for fn in handlers:
        try:
            # metrics must be dispatched or offloaded within 100ms
            # or the metrics handler will be deregistered
            with timeout(seconds=0.100):
                fn(f"modin.{name}", value)
        except Exception:
            clear_metric_handler(fn)


def add_metric_handler(handler: Callable[[str, Union[int, float]], None]) -> None:
    if MetricsMode.get() == "disable":
        return
    _metric_handlers.append(handler)


def clear_metric_handler(handler) -> None:
    if handler in _metric_handlers:
        _metric_handlers.remove(handler)
