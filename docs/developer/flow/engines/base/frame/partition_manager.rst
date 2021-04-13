BaseFrameManager
""""""""""""""""

The class is base for any partition manager class of ``pandas`` backend and serves as
intermediate level between ``pandas`` base frame and conforming partition class.
The class is responsible for partitions manipulation and applying a function to individual partitions:
block partitions, row partitions or column partitions, i.e. the class can form axis partitions from block partitions
to apply a function if an operation requires access to an entire column or row. The class translates frame API
into partition API and also can have some preprocessing operations in depend of the partition type
for performance improving (for example, ``preprocess_func``):

API
---

.. autoclass:: modin.engines.base.frame.partition_manager.BaseFrameManager
  :noindex:
  :members:
