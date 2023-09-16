:orphan:

ExperimentalPandasOnUnidist Execution
=====================================

`ExperimentalPandasOnUnidist` execution keeps the underlying mechanisms of :doc:`PandasOnUnidist </flow/modin/core/execution/unidist/implementations/pandas_on_unidist/index>`
execution architecturally unchanged and adds experimental features of ``Data Transformation``, ``Data Ingress`` and ``Data Egress`` (e.g. :py:func:`~modin.experimental.pandas.read_pickle_distributed`).

PandasOnUnidist and ExperimentalPandasOnUnidist differences
-----------------------------------------------------------

- another Factory ``PandasOnUnidistFactory`` -> ``ExperimentalPandasOnUnidistFactory``
- another IO class ``PandasOnUnidistIO`` -> ``ExperimentalPandasOnUnidistIO``

ExperimentalPandasOnUnidistIO classes and modules
-------------------------------------------------

- :py:class:`~modin.experimental.core.execution.unidist.implementations.pandas_on_unidist.io.io.ExperimentalPandasOnUnidistIO`
- :py:class:`~modin.core.execution.dispatching.factories.factories.ExperimentalPandasOnUnidistFactory`
- :py:class:`~modin.experimental.core.io.text.csv_glob_dispatcher.ExperimentalCSVGlobDispatcher`
- :py:class:`~modin.experimental.core.io.sql.sql_dispatcher.ExperimentalSQLDispatcher`
- :py:class:`~modin.experimental.core.io.pickle.pickle_dispatcher.ExperimentalPickleDispatcher`
- :py:class:`~modin.experimental.core.io.text.custom_text_dispatcher.ExperimentalCustomTextDispatcher`
- :py:class:`~modin.core.storage_formats.pandas.parsers.PandasCSVGlobParser`
- :doc:`ExperimentalPandasOnUnidist IO module </flow/modin/experimental/core/execution/unidist/implementations/pandas_on_unidist/io/index>`
