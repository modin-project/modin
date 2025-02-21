Modin Metrics
=============

Modin allows for third-party systems to register a metrics handler to collect specific API statistics.
Metrics have a name and a value, can be aggregated, discarded, or emitted without impact to the program.

CPU load, memory usage, and disk usage are all typical metrics; but for modin metrics are current 
collected on API timings which can be used to optimize end-user interactive performance. New metrics may 
be added in the future.

It is the responsibility of the handler to process or forward these metrics. The name of the metric must 
be in "dot format" and all lowercase, similar to graphite or rrd. The value is an integer or float.

Handlers are functions of the form: `fn(str, int|float)` and can be registered with:

.. code-block:: python

  import modin.pandas as pd
  from modin.logging.metrics import add_metric_handler

  def func(name: str, value: int | float):
    print(f"Got metric {name} value {value}")

  add_metric_handler(func)

.. warning:: 
  A metric handler should be non-blocking, returning within 100ms, although this is not enforced. It must not throw exceptions or it will
  be deregistered. These restrictions are to help guard against the implementation of a metrics collector which would impact
  interactice performance significantly. The data from metrics should generally be offloaded to another system for processing
  and not involve any direct network calls.

Disable Modin metrics like so:
.. code-block:: python

  import modin.pandas as pd
  from modin.config import MetricsMode
  MetricsMode.disable()
