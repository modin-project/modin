:orphan:

PyArrow-on-Ray Module Description
"""""""""""""""""""""""""""""""""

High-Level Module Overview
''''''''''''''''''''''''''

This module houses experimental functionality with PyArrow backend and Ray
engine. The biggest difference from core engines is that internally each partition
is represented as ``pyarrow.Table`` put in the ``Ray`` Plasma store.

Why to Use PyArrow Tables
'''''''''''''''''''''''''

As it was `mentioned <https://wesmckinney.com/blog/apache-arrow-pandas-internals/>`_
by the pandas creator, pandas internal architecture is not optimal and sometimes
needs up to ten times more memory than the original dataset size
(note, that pandas rule of thumb: `have 5 to 10 times as much RAM as the size of your
dataset`). In order to fix this issue (or at least to reduce needed memory amount and
needed data copying), ``PyArrow-on-Ray`` module was added. Due to optimized architecture
of PyArrow Tables, number of needed copies can be decreased `down to zero
<https://arrow.apache.org/docs/python/pandas.html#zero-copy-series-conversions>`_ in some
corner cases, that can signifficantly improve Modin performance. The downside of this approach
is that PyArrow and pandas do not support the same APIs and some functions/parameters can have
incompatibilities or output different results, so for now ``PyArrow-on-Ray`` engine is
under development and marked as experimental.
