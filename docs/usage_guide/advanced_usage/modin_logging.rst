Modin Logging
=============

Modin logging offers users greater insight into their queries by logging internal Modin API calls, partition metadata, 
and profiling system memory. When Modin logging is enabled (default disabled), log files are written to a local `.modin` directory at the same
directory level as the notebook/script used to run Modin. It is possible to configure whether to log system memory and additional metadata 
in addition to Modin API calls (see the usage examples below).

The logs that contain the Modin API stack traces are named `job_<job_uuid>.log`. The logs that contain the memory utilization metrics are 
named `memory_<job_uuid>.log`. If any log file exceeds 10MB, the logger will append an integer to the log name. For instance if you have 
20MB worth of Modin API logs, you can expect to find `job_<job_uuid>.log.1` and `job_<job_uuid>.log.2` in the `.modin` directory.

**Developer Warning:** In some cases, running Modin logging inside of a Jupyterlab or other notebook instance at the root level of the Modin 
repository or in the `modin/modin` directory may result in circular dependency issues. This is as a result of a name conflict between the 
`modin/logging` directory and the Python `logging` module, which may be used as a default in such environments. As a resolution, one can simply
run Modin logging from a different directory or manually manage the name conflicts.

Usage examples
--------------

In the example below, we enable logging for internal Modin API calls. 

.. code-block:: python

  import modin.pandas as pd
  from modin.config import LogMode
  LogMode.enable_api_only()

  # Your code goes here

In the next example, we add logging for not only internal Modin API calls, but also for partition metadata and memory profiling.
We can set the granularity (in seconds) at which the system memory utilization is logged using `LogMemoryInterval`. 
We can also set the size of the memory logs (in MBs) using `LogMemorySize`. 

.. code-block:: python

  import modin.pandas as pd
  from modin.config import LogMode, LogMemoryInterval, LogMemorySize 
  LogMode.enable()
  LogMemoryInterval.put(2) # Defaults to 5 seconds, new interval is 2 seconds
  LogMemorySize.put(5) # Defaults to 10 MB per log file, new size is 5 MB 

  # Your code goes here

Disable Modin logging like so:

.. code-block:: python

  import modin.pandas as pd
  from modin.config import LogMode
  LogMode.disable()

  # Your code goes here
