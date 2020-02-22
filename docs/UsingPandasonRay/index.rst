Ray
===

This section describes usage related documents for the Pandas on Ray component of Modin.

Modin uses Pandas on Ray by default, but if you wanted to be explicit, you could set the
following environment variables:

.. code-block:: bash

   export MODIN_ENGINE=ray
   export MODIN_BACKEND=pandas
