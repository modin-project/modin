:orphan:

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
  * FIX-#3368: support unsigned integers in OmniSci backend (#4256)
  * FIX-#4057: Allow reading an empty parquet file (#4075)
  * FIX-#3884: Fix read_excel() dropping empty rows (#4161)
  * FIX-#4257: Fix Categorical() for scalar categories (#4258)
  * FIX-#4300: Fix Modin Categorical column dtype categories (#4276)
  * FIX-#4208: Fix lazy metadata update for `PandasDataFrame.from_labels` (#4209)
  * FIX-#3981, FIX-#3801, FIX-#4149: Stop broadcasting scalars to set items (#4160)
  * FIX-#4185: Fix rolling across column partitions (#4262)
  * FIX-#4303: Fix the syntax error in reading from postgres (#4304)
  * FIX-#4308: Add proper error handling in df.set_index (#4309)
  * FIX-#4056: Allow an empty parse_date list in `read_csv_glob` (#4074)
  * FIX-#4312: Fix constructing categorical frame with duplicate column names (#4313).
  * FIX-#4314: Allow passing a series of dtypes to astype (#4318)
  * FIX-#4310: Handle lists of lists of ints in read_csv_glob (#4319)
* Performance enhancements
  * FIX-#4138, FIX-#4009: remove redundant sorting in the internal '.mask()' flow (#4140)
  * FIX-#4183: Stop shallow copies from creating global shared state. (#4184)
* Benchmarking enhancements
  * FIX-#4221: add `wait` method for `PandasOnRayDataframeColumnPartition` class (#4231)
* Refactor Codebase
  * REFACTOR-#3990: remove code duplication in `PandasDataframePartition` hierarchy (#3991)
  * REFACTOR-#4229: remove unused `dask_client` global variable in `modin\pandas\__init__.py` (#4230)
  * REFACTOR-#3997: remove code duplication for `broadcast_apply` method (#3996)
  * REFACTOR-#3994: remove code duplication for `get_indices` function (#3995)
  * REFACTOR-#4331: remove code duplication for `to_pandas`, `to_numpy` functions in `QueryCompiler` hierarchy (#4332)
  * REFACTOR-#4213: Refactor `modin/examples/tutorial/` directory (#4214)
  * REFACTOR-#4206: add assert check into `__init__` method of `PandasOnDaskDataframePartition` class (#4207)
  * REFACTOR-#3900: add flake8-no-implicit-concat plugin and refactor flake8 error codes (#3901)
  * REFACTOR-#4093: Refactor base to be smaller (#4220)
  * REFACTOR-#4047: Rename `cluster` directory to `cloud` in examples (#4212)
  * REFACTOR-#3853: interacting with Dask interface through `DaskWrapper` class (#3854)
  * REFACTOR-#4322: Move is_reduce_fn outside of groupby_agg (#4323)
* Pandas API implementations and improvements
  * FEAT-#3603: add experimental `read_custom_text` function that can read custom line-by-line text files (#3441)
  * FEAT-#979: Enable reading from SQL server (#4279)
* OmniSci enhancements
  *
* XGBoost enhancements
  *
* Developer API enhancements
  * FEAT-#4245: Define base interface for dataframe exchange protocol (#4246)
  * FEAT-#4244: Implement dataframe exchange protocol for HdkOnNative execution (#4269)
  * FEAT-#4144: Implement dataframe exchange protocol for pandas storage format (#4150)
  * FEAT-#4342: Support `from_dataframe`` for pandas storage format (#4343)
* Update testing suite
  * TEST-#3628: Report coverage data for `test-internals` CI job (#4198)
  * TEST-#3938: Test tutorial notebooks in CI (#4145)
  * TEST-#4153: Fix condition of running lint-commit and set of CI triggers (#4156)
  * TEST-#4201: Add read_parquet, explode, tail, and various arithmetic functions to asv_bench (#4203)
* Documentation improvements
  * DOCS-#4077: Add release notes template to docs folder (#4078)
  * DOCS-#4082: Add pdf/epub/htmlzip formats for doc builds (#4083)
  * DOCS-#4168: Fix rendering the examples on troubleshooting page (#4169)
  * DOCS-#4151: Add info in troubleshooting page related to Dask engine usage (#4152)
  * DOCS-#4172: Refresh Intel Distribution of Modin paragraph (#4175)
  * DOCS-#4173: Mention strict channel priority in conda install section (#4178)
  * DOCS-#4176: Update OmniSci usage section (#4192)
  * DOCS-#4027: Add GIF images and chart to Modin README demonstrating speedups (#4232)
  * DOCS-#3954: Add Dask example notebooks (#4139)
  * DOCS-#4272: Add bar chart comparisons to quick start guide (#4277)
  * DOCS-#3953: Add docs and notebook examples on running Modin with OmniSci (#4001)
  * DOCS-#4280: Change links in jupyter notebooks (#4281)
  * DOCS-#4290: Add changes for OmniSci notebooks (#4291)
  * DOCS-#4241: Update warnings and docs regarding defaulting to pandas (#4242)
  * DOCS-#3099: Fix `BasePandasDataSet` docstrings warnings (#4333)
  * DOCS-#4339: Reformat I/O functions docstrings (#4341)
  * DOCS-#4336: Reformat general utilities docstrings (#4338)
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
@jeffreykennethli
@dorisjlee
@Rubtsowa
