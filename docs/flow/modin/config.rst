:orphan:

Config Module Overview
""""""""""""""""""""""

Using this module, the user can tune Modin's behavior. To see
all avaliable configs just run `python -m modin.config`, this command will print all
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

See example of interation with Modin configs below, as it can be seen config
value can be set either by setting the environment variable or by using config
API.

.. code-block:: python

    import os

    # Setting `MODIN_STORAGE_FORMAT` environment variable.
    # Also can be set outside the script.
    os.environ["MODIN_STORAGE_FORMAT"] = "OmniSci"

    import modin.config
    import modin.pandas as pd

    # Checking initially set `StorageFormat` config,
    # which corresponds to `MODIN_STORAGE_FORMAT` environment
    # variable
    print(modin.config.StorageFormat.get()) # prints 'Omnisci'

    # Checking default value of `NPartitions`
    print(modin.config.NPartitions.get()) # prints '8'

    # Changing value of `NPartitions`
    modin.config.NPartitions.put(16)
    print(modin.config.NPartitions.get()) # prints '16'
