:orphan:

Modin 0.15.0

Key Features and Updates
------------------------

* Stability and Bugfixes
  * FIX-#4376: Upgrade pandas to 1.4.2 (#4377)
  * FIX-#3615: Relax some deps in development env (#4365)
  * FIX-#4370: Fix broken docstring links (#4375)
  * FIX-#4392: Align Modin XGBoost with xgb>=1.6 (#4393)
  * FIX-#3527: Fix parquet partitioning issue causing negative row length partitions (#4368)
* Performance enhancements
  * FEAT-#4320: Add connectorx as an alternative engine for read_sql (#4346)
* Benchmarking enhancements
  *
* Refactor Codebase
  *
* Pandas API implementations and improvements
  *
* OmniSci enhancements
  *
* XGBoost enhancements
  *
* Developer API enhancements
  * FEAT-#4359: Add __dataframe__ method to the protocol dataframe (#4360)
* Update testing suite
  * TEST-#4363: Use Ray from pypi in CI (#4364)
* Documentation improvements
  * DOCS-#4296: Fix docs warnings (#4297)
  * DOCS-#4388: Turn off fail_on_warning option for docs build (#4389)
* Dependencies
  * FIX-#4327: Update min pin for xgboost version (#4328)
  * FIX-#4383: Remove `pathlib` from deps (#4384)
  * FIX-#4390: Add `redis` to Modin dependencies (#4396)

Contributors
------------
@YarShev
@Garra1980
@prutskov
@alexander3774
@amyskov
@wangxiaoying
@jeffreykennethli