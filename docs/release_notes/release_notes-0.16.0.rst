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
  * FIX-#4564: Workaround import issues in Ray: auto-import pandas on python start if env var is set (#4603)
* Performance enhancements
  * PERF-#4182: Add cell-wise execution for binary ops, fix bin ops for empty dataframes (#4391)
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
* Documentation improvements
  * DOCS-#4552: Change default sphinx language to en to fix sphinx >= 5.0.0 build (#4553)
* Dependencies
  * FEAT-#4598: Add support for pandas 1.4.3 (#4599)
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
