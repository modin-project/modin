:orphan:

Experimental Modules Overview
"""""""""""""""""""""""""""""
In some cases Modin can give user the opportunity to extend (not modify) typical pandas
API or to try not compeletely tested functionality in order to get more flexibility or
to increase the performance gain. In that cases, depending on exact experimental feature,
user will need to install additional packages, set environment variables or replace standard
Modin import statement ``import modin.pandas as pd`` with modified version
``import modin.experimental.pandas as pd``. For concreate experimental feature example, please
refer to the needed link from the :ref:`directory tree <directory-tree>`.
