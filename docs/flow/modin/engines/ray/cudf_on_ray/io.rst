:orphan:

IO details in cuDF backend
""""""""""""""""""""""""""

IO on cuDF backend is implemented using base classes ``BaseIO`` and ``CSVDispatcher``.

cuDFOnRayIO
"""""""""""

The class ``cuDFOnRayIO`` implements ``BaseIO`` base class using cuDF-backend
entities (``cuDFOnRayFrame``, ``cuDFOnRayFramePartition`` etc.).

Public API
----------

.. autoclass:: modin.engines.ray.cudf_on_ray.io.io.cuDFOnRayIO
  :noindex:
  :members:


cuDFCSVDispatcher
"""""""""""""""""

The ``cuDFCSVDispatcher`` class implements ``CSVDispatcher`` using cuDF backend.

.. autoclass:: modin.engines.ray.cudf_on_ray.io.text.csv_dispatcher.cuDFCSVDispatcher
  :noindex:
  :members: