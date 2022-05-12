Logging and Tracing with Modin
==============================

Modin logging and tracing allows users to gain observability into their queries by logging internal Modin API calls, partition metadata, 
and memory profiling. When Modin logging is enabled (default disabled), log files are written to a local `.modin` directory at the same
directory level as the notebook/script used to run Modin. Each log file in the `.modin` directory is named after the job uuid. 

Usage example
-------------

In the example below, we enable logging for internal Modin API calls. 

.. code-block:: python

  import modin.pandas as pd
  from modin.config import LogMode

  LogMode.enable_api_only()

  # Your code goes here

In this example, we enable logging for internal Modin API calls, partition metadata, and memory profiling.
We can set the granularity (in seconds) at which the system memory utilization is logged using `LogMemoryInterval`. 

.. code-block:: python

  import modin.pandas as pd
  from modin.config import LogMode, LogMemoryInterval

  LogMode.enable()
  LogMemoryInterval(2) # Defaults to 5 seconds

  # Your code goes here

Modin logging is default disabled, but if currently enabled you can manually disable the logs as follows:

.. code-block:: python

  import modin.pandas as pd
  from modin.config import LogMode
  LogMode.disable()

  # Your code goes here
