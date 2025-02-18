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


def clear_metric_handler(handler: Callable[[str, Union[int, float]], None]) -> None:
    if handler in _metric_handlers:
        _metric_handlers.remove(handler)
