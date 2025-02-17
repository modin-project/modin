import logging
import re
from typing import Callable

from modin.config.envvars import MetricsMode
from modin.utils import timeout


metric_name_pattern = r"[a-zA-Z\.]+$"
_metric_handlers = []

# Metric/Telemetry hooks can be implemented by plugin engines
# to collect discrete data on how modin is performing at the
# high level modin layer.
def emit_metric(name:str, value:int|float):
    ''' 
    emit a metric using the set of registered handlers
    '''
    if MetricsMode.get() == "disable":
        return
    if not re.fullmatch(metric_name_pattern, name):
        raise KeyError(f"Metrics name is not in metric-name dot format, (eg. modin.dataframe.hist.duration ): {name}")
    

    handlers = _metric_handlers.copy()
    for fn in handlers:
        try:
            with timeout(seconds=1):
                fn(f"modin.{msg}", value)
        except:
            logging.ERROR("telemetry handler threw exception, removing handler: " + e)
            _metric_handlers.remove(fn)
    pass

def add_metric_handler(handler:Callable[[str, int|float], None]):
    _metric_handlers.append(handler)
    
def clear_metric_handler(handler):
    _telemetry_handlers = []
    
    