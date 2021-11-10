:orphan:

IO details in cuDFOnRay execution
"""""""""""""""""""""""""""""""""

IO on cuDFOnRay execution is implemented using base classes ``BaseIO`` and ``CSVDispatcher``.

cuDFOnRayIO
"""""""""""

The class ``cuDFOnRayIO`` implements ``BaseIO`` base class using cuDFOnRay-execution
entities (``cuDFOnRayDataframe``, ``cuDFOnRayDataframePartition`` etc.).

Public API
----------

.. autoclass:: modin.core.execution.ray.implementations.cudf_on_ray.io.io.cuDFOnRayIO
  :noindex:
  :members:


cuDFCSVDispatcher
"""""""""""""""""

The ``cuDFCSVDispatcher`` class implements ``CSVDispatcher`` using cuDFOnRay execution.

.. autoclass:: modin.core.execution.ray.implementations.cudf_on_ray.io.text.csv_dispatcher.cuDFCSVDispatcher
  :noindex:
  :members: