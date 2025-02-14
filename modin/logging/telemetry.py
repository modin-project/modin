import logging
from modin.utils import timeout

_telemetry_handlers = []

# Telemetry hooks can be implemented by plugin engines
# to collect discrete data on how modin is performing at the
# high level moden layer.
def emit_telemetry_event(msg:str, value:int|float):
    _telemetry_handlers
    handlers = _telemetry_handlers.copy()
    for fn in handlers:
        try:
            with timeout(seconds=1):
                fn(f"modin::{msg}", value)
        except:
            logging.ERROR("telemetry handler threw exception, removing handler: " + e)
            _telemetry_handlers.remove(fn)
    pass

def add_telemetry_handler(handler:callable):
    _telemetry_handlers.append(handler)
    
def clear_telemetry_handler(handler):
    _telemetry_handlers = []
    
    