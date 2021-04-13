BasePandasFrame
"""""""""""""""

The class is base for any frame class of ``pandas`` backend and serves as the intermediate level
between ``pandas`` query compiler and conforming partition manager. All queries formed
at the query compiler layer are ingested by this class and then conveyed jointly with the stored partitions
into the partition manager for processing. Direct partitions manipulation by this class is prohibited except
cases if an operation is striclty private or protected and called inside of the class only. The class provides
significantly reduced set of operations that fit plenty of pandas operations.

API
---

.. autoclass:: modin.engines.base.frame.data.BasePandasFrame
  :noindex:
  :members:
