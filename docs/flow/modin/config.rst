:orphan:

Modin Configuration Settings
""""""""""""""""""""""""""""

To adjust Modin's default behavior, you can set the value of Modin
configs by setting an environment variable or by using the
``modin.config`` API. To list all available configs in Modin, please
run ``python -m modin.config`` to print all
Modin configs with descriptions.

Public API
''''''''''

Potentially, the source of configs can be any, but for now only environment
variables are implemented. Any environment variable originate from
:class:`~modin.config.envvars.EnvironmentVariable`, which contains most of
the config API implementation.

.. autoclass:: modin.config.envvars.EnvironmentVariable
  :members: get, put, get_help, get_value_source, once, subscribe

Modin Configs List
''''''''''''''''''

.. csv-table::
   :file: configs_help.csv
   :header-rows: 1

Usage Guide
'''''''''''

See example of interaction with Modin configs below, as it can be seen config
value can be set either by setting the environment variable or by using config
API.

.. code-block:: python

    import os

    # Setting `MODIN_ENGINE` environment variable.
    # Also can be set outside the script.
    os.environ["MODIN_ENGINE"] = "Dask"

    import modin.config
    import modin.pandas as pd

    # Checking initially set `Engine` config,
    # which corresponds to `MODIN_ENGINE` environment
    # variable
    print(modin.config.Engine.get()) # prints 'Dask'

    # Checking default value of `NPartitions`
    print(modin.config.NPartitions.get()) # prints '8'

    # Changing value of `NPartitions`
    modin.config.NPartitions.put(16)
    print(modin.config.NPartitions.get()) # prints '16'

One can also use config variables with a context manager in order to use
some config only for a certain part of the code:

.. code-block:: python

    import modin.config as cfg

    # Default value for this config is 'False'
    print(cfg.RangePartitioning.get()) # False

    # Set the config to 'True' inside of the context-manager
    with cfg.context(RangePartitioning=True):
        print(cfg.RangePartitioning.get()) # True
        df.merge(...) # will use range-partitioning impl

    # Once the context is over, the config gets back to its previous value
    print(cfg.RangePartitioning.get()) # False

    # You can also set multiple config at once when you pass a dictionary to 'cfg.context'
    print(cfg.AsyncReadMode.get()) # False

    with cfg.context(RangePartitioning=True, AsyncReadMode=True):
        print(cfg.RangePartitioning.get()) # True
        print(cfg.AsyncReadMode.get()) # True
    print(cfg.RangePartitioning.get()) # False
    print(cfg.AsyncReadMode.get()) # False
