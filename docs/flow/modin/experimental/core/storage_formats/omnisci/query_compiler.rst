DFAlgQueryCompiler
""""""""""""""""""

:py:class:`~modin.experimental.core.storage_formats.omnisci.query_compiler.DFAlgQueryCompiler` implements
a query compiler for lazy frame. Each compiler instance holds an instance of
:py:class:`~modin.experimental.core.execution.native.implementations.omnisci_on_native.dataframe.dataframe.OmnisciOnNativeDataframe`
which is used to build a lazy execution tree.

Public API
''''''''''

.. autoclass:: modin.experimental.core.storage_formats.omnisci.query_compiler.DFAlgQueryCompiler
  :members:
