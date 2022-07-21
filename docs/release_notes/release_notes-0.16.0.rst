:orphan:

Modin 0.16.0

Key Features and Updates
------------------------

* Stability and Bugfixes
  * FIX-#4570: Replace ``np.bool`` -> ``np.bool_`` (#4571)
  * FIX-#4543: Fix `read_csv` in case skiprows=<0, []> (#4544)
  * FIX-#4059: Add cell-wise execution for binary ops, fix bin ops for empty dataframes (#4391)
  * FIX-#4589: Pin protobuf<4.0.0 to fix ray (#4590)
  * FIX-#4577: Set attribute of Modin dataframe to updated value (#4588)
  * FIX-#4411: Fix binary_op between datetime64 Series and pandas timedelta (#4592)
  * FIX-#4604: Fix `groupby` + `agg` in case when multicolumn can arise (#4642)
  * FIX-#4582: Inherit custom log layer (#4583)
  * FIX-#4639: Fix `storage_options` usage for `read_csv` and `read_csv_glob` (#4644)
  * FIX-#4593: Ensure Modin warns when setting columns via attributes (#4621)
  * FIX-#4584: Enable pdb debug when running cloud tests (#4585)
  * FIX-#4564: Workaround import issues in Ray: auto-import pandas on python start if env var is set (#4603)
  * FIX-#4641: Reindex pandas partitions in `df.describe()` (#4651)
  * FIX-#2064: Fix `iloc`/`loc` assignment when dataframe is empty (#4677)
  * FIX-#4634: Check for FrozenList as `by` in `df.groupby()` (#4667)
  * FIX-#4491: Wait for all partitions in parallel in benchmark mode (#4656)
  * FIX-#4358: MultiIndex `loc` shouldn't drop levels for full-key lookups (#4608)
* Performance enhancements
  * PERF-#4182: Add cell-wise execution for binary ops, fix bin ops for empty dataframes (#4391)
  * PERF-#4288: Improve perf of `groupby.mean` for narrow data (#4591)
  * PERF-#4325: Improve perf of multi-column assignment in `__setitem__` when no new column names are assigning (#4455)
* Benchmarking enhancements
  *
* Refactor Codebase
  * REFACTOR-#4530: Standardize access to physical data in partitions (#4563)
  * REFACTOR-#4534: Replace logging meta class with class decorator (#4535)
* Pandas API implementations and improvements
  * FEAT-#4670: Implement convert_dtypes by mapping across partitions (#4671)
* OmniSci enhancements
  *
* XGBoost enhancements
  *
* Developer API enhancements
  *
* Update testing suite
  * TEST-#4508: Reduce test_partition_api pytest threads to deflake it (#4551)
  * TEST-#4550: Use much less data in test_partition_api (#4554)
  * TEST-#4610: Remove explicit installation of `black`/`flake8` for omnisci ci-notebooks (#4609)
  * TEST-#2564: Add caching and use mamba for conda setups in GH (#4607)
  * TEST-#4557: Delete multiindex sorts instead of xfailing (#4559)  
* Documentation improvements
  * DOCS-#4552: Change default sphinx language to en to fix sphinx >= 5.0.0 build (#4553)
  * DOCS-#4628: Add to_parquet partial support notes (#4648)
* Dependencies
  * FEAT-#4598: Add support for pandas 1.4.3 (#4599)
  * FEAT-#4619: Integrate mypy static type checking (#4620)
* New Features
  * FEAT-4463: Add experimental fuzzydata integration for testing against a randomized dataframe workflow (#4556)
  * FEAT-#4419: Extend virtual partitioning API to pandas on Dask (#4420)
  * FEAT-#4147: Add partial compatibility with Python 3.6 and pandas 1.1 (#4301)

Contributors
------------
@mvashishtha
@NickCrews
@prutskov
@vnlitvinov
@pyrito
@suhailrehman
@RehanSD
@helmeleegy
@anmyachev
@d33bs
@noloerino
@devin-petersohn
@YarShev
