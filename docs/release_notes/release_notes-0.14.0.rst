Modin 0.14.0

Key Features and Updates
------------------------

* Stability and Bugfixes
  * FIX-#4058: Allow pickling empty dataframes and series (#4095)
  * FIX-#4136: Fix exercise_3.ipynb example notebook (#4137)
  * FIX-#4105: Fix names of pandas options to avoid `OptionError` (#4109)
  * FIX-#3417: Fix read_csv with skiprows and header parameters (#3419)
  * FIX-#4142: Fix OmniSci enabling (#4146)
  * FIX-#4162: Use `skipif` instead of `skip` for compatibility with pytest 7.0 (#4163)
  * FIX-#4158: Do not print OmniSci logs to stdout by default (#4159)
  * FIX-#4177: Support read_feather from pathlike objects (#4177)
  * FIX-#4234: Upgrade pandas to 1.4.1 (#4235)
  * FIX-#4241: Update warnings and docs regarding defaulting to pandas (#4242)
  * FIX-#4057: Allow reading an empty parquet file (#4075)  
  * FIX-#3884: Fix read_excel() dropping empty rows (#4161)
* Performance enhancements
  * FIX-#4138, FIX-#4009: remove redundant sorting in the internal '.mask()' flow (#4140)
* Benchmarking enhancements
  *
* Refactor Codebase
  * REFACTOR-#3990: remove code duplication in `PandasDataframePartition` hierarchy (#3991)
  * REFACTOR-#4229: remove unused `dask_client` global variable in `modin\pandas\__init__.py` (#4230)
  * REFACTOR-#3997: remove code duplication for `broadcast_apply` method (#3996)
  * REFACTOR-#3994: remove code duplication for `get_indices` function (#3995)
  * REFACTOR-#4213: Refactor `modin/examples/tutorial/` directory (#4214)
  * REFACTOR-#4206: add assert check into `__init__` method of `PandasOnDaskDataframePartition` class (#4207)
* Pandas API implementations and improvements
  *
* OmniSci enhancements
  *
* XGBoost enhancements
  *
* Developer API enhancements
  *
* Update testing suite
  * TEST-#3628: Report coverage data for `test-internals` CI job (#4198)
  * TEST-#3938: Test tutorial notebooks in CI (#4145)
  * TEST-#4153: Fix condition of running lint-commit and set of CI triggers (#4156)
* Documentation improvements
  * DOCS-#4077: Add release notes template to docs folder (#4078)
  * DOCS-#4082: Add pdf/epub/htmlzip formats for doc builds (#4083)
  * DOCS-#4168: Fix rendering the examples on troubleshooting page (#4169)
  * DOCS-#4151: Add info in troubleshooting page related to Dask engine usage (#4152)
  * DOCS-#4172: Refresh Intel Distribution of Modin paragraph (#4175)
  * DOCS-#4173: Mention strict channel priority in conda install section (#4178)
  * DOCS-#4176: Update OmniSci usage section (#4192)
* Dependencies
  * FIX-#4113, FIX-#4116, FIX-#4115: Apply new `black` formatting, fix pydocstyle check and readthedocs build (#4114)
  * TEST-#3227: Use codecov github action instead of bash form in GA workflows (#3226)
  * FIX-#4115: Unpin `pip` in readthedocs deps list (#4170)
  * TEST-#4217: Pin `Dask<2022.2.0` as a temporary fix of CI (#4218)

Contributors
------------
@prutskov
@amyskov
@paulovn
@anmyachev
@YarShev
@RehanSD
@devin-petersohn
@dchigarev
@Garra1980
@mvashishtha
@naren-ponder
