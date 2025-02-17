Modin Metrics
=============

Modin allows for third-party systems to register a metrics handler to collect specific API statistics.
it is the responsibility of the handler to process or forward these metrics. Each metric is collected
with a name and value. The name of the metric must be in "dot format" and all lowercase. The value is
an integer or float.

Handlers are functions of the form: `fn(str, int|float)` and can be registered with:

.. code-block:: python

  import modin.pandas as pd
  from modin.logging.metrics import add_metric_handler

  def func(name:str, value:int:float):
    print(f"Got metric {name} value {value}")

  add_metric_handler(func)

.. warning:: 
  A metric handler must return within 100ms or it will be disabled and deregistered. It must not throw exceptions or it will
  be deregistered. These restrictions are to help guard against the implementation of a metrics collector which would impact
  interactice performance significantly. The data from metrics should generally be offloaded to another system for processing
  and not involve any direct network calls.

Disable Modin metrics like so:
.. code-block:: python

  import modin.pandas as pd
  from modin.config import MetricsMode
  MetricsMode.disable()
