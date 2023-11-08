DFAlgQueryCompiler
""""""""""""""""""

:py:class:`~modin.experimental.core.storage_formats.hdk.query_compiler.DFAlgQueryCompiler` implements
a query compiler for lazy frame. Each compiler instance holds an instance of
:py:class:`~modin.experimental.core.execution.native.implementations.hdk_on_native.dataframe.dataframe.HdkOnNativeDataframe`
which is used to build a lazy execution tree.

Public API
''''''''''

.. autoclass:: modin.experimental.core.storage_formats.hdk.query_compiler.DFAlgQueryCompiler
  :members:
