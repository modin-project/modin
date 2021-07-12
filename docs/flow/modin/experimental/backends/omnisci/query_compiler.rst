OmniSci Query Compiler
""""""""""""""""""""""
:py:class:`~modin.experimental.backends.omnisci.query_compiler.DFAlgQueryCompiler` implements
a query compiler for lazy frame. Each compiler instance holds an instance of
:py:class:`~modin.experimental.engines.omnisci_on_ray.frame.data.OmnisciOnRayFrame`
which is used to build a lazy execution tree.

Public API
''''''''''

.. autoclass:: modin.experimental.backends.omnisci.query_compiler.DFAlgQueryCompiler
  :members:
