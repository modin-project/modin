:orphan:

Experimental Modules Overview
"""""""""""""""""""""""""""""

In some cases Modin can give the user the opportunity to extend (not modify) typical pandas
API or to try new functionality in order to get more flexibility. Depending on the exact
experimental feature user may need to install additional packages, change configurations or
replace the standard Modin import statement ``import modin.pandas as pd`` with modified version
``import modin.experimental.pandas as pd``.

``modin.experimental`` holds experimental functionality that is under development right now
and provides a limited set of functionality:

* :doc:`xgboost <xgboost>`
* :doc:`sklearn <sklearn>`
* :doc:`batch <batch>`


.. toctree::
    :hidden:

    sklearn
    xgboost
    batch
