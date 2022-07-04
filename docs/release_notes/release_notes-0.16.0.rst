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
  * FIX-#4582: Inherit custom log layer (#4583)
  * FIX-#4593: Ensure Modin warns when setting columns via attributes (#4621)
  * FIX-#4584: Enable pdb debug when running cloud tests (#4585)
* Performance enhancements
  * PERF-#4182: Add cell-wise execution for binary ops, fix bin ops for empty dataframes (#4391)
  * PERF-#4288: Improve perf of `groupby.mean` for narrow data (#4591)
* Benchmarking enhancements
  *
* Refactor Codebase
  * REFACTOR-#4530: Standardize access to physical data in partitions (#4563)
  * REFACTOR-#4534: Replace logging meta class with class decorator (#4535)
* Pandas API implementations and improvements
  *
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
* Documentation improvements
  * DOCS-#4552: Change default sphinx language to en to fix sphinx >= 5.0.0 build (#4553)
* Dependencies
  * FEAT-#4598: Add support for pandas 1.4.3 (#4599)
  * FEAT-#4619: Integrate mypy static type checking (#4620)
* New Features
  * FEAT-4463: Add experimental fuzzydata integration for testing against a randomized dataframe workflow (#4556)

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
