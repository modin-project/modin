Modin Logging
=============

Modin logging offers users greater insight into their queries by logging internal Modin API calls, partition metadata, 
and profiling system memory. When Modin logging is enabled (default disabled), log files are written to a local `.modin` directory at the same
directory level as the notebook/script used to run Modin. Each log file in the `.modin` directory is uniquely named after the job uuid. 

**Developer Warning:** In some cases, running Modin logging inside of a Jupyterlab or other notebook instance at the root level of the Modin 
repository or in the `modin/modin` directory may result in circular dependency issues. This is as a result of a name conflict between the 
`modin/logging` directory and the Python `logging` module, which may be used as a default in such environments. As a resolution, one can simply
run Modin logging from a different directory or manually manage the name conflicts.


Usage example
-------------

In the example below, we enable logging for internal Modin API calls. 

.. code-block:: python

  import modin.pandas as pd
  from modin.config import LogMode
  LogMode.enable_api_only()

  # Your code goes here

In the next example, we add logging for not only internal Modin API calls, but also for partition metadata and memory profiling.
We can set the granularity (in seconds) at which the system memory utilization is logged using `LogMemoryInterval`. 

.. code-block:: python

  import modin.pandas as pd
  from modin.config import LogMode, LogMemoryInterval
  LogMode.enable()
  LogMemoryInterval.put(2) # Defaults to 5 seconds, new interval is 2 seconds

  # Your code goes here

Disable Modin logging like so:

.. code-block:: python

  import modin.pandas as pd
  from modin.config import LogMode
  LogMode.disable()

  # Your code goes here
