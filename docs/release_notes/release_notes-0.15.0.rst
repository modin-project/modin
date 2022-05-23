:orphan:

Modin 0.15.0

Key Features and Updates
------------------------

* Stability and Bugfixes
  * FIX-#4376: Upgrade pandas to 1.4.2 (#4377)
  * FIX-#3615: Relax some deps in development env (#4365)
  * FIX-#4370: Fix broken docstring links (#4375)
  * FIX-#4392: Align Modin XGBoost with xgb>=1.6 (#4393)
  * FIX-#4385: Get rid of `use-deprecated` option in `pip` (#4386)
  * FIX-#3527: Fix parquet partitioning issue causing negative row length partitions (#4368)
  * FIX-#4330: Override the memory limit to start ray 1.11.0 on Macs (#4335)
  * FIX-#4407: Align `insert` function with pandas in case of numpy array with several columns (#4408)
  * FIX-#4373: Fix invalid file path when trying `read_csv_glob` with `usecols` parameter (#4405)
  * FIX-#4394: Fix issue with multiindex metadata desync (#4395)
  * FIX-#4425: Add parameters to groupby pct_change (#4429)
  * FIX-#4414: Add missing f prefix on f-strings found at https://codereview.doctor (#4415)
  * FIX-#4461: Fix S3 CSV data path (#4462)
* Performance enhancements
  * FEAT-#4320: Add connectorx as an alternative engine for read_sql (#4346)
* Benchmarking enhancements
  *
* Refactor Codebase
  * REFACTOR-#4284: use variable length unpacking when getting results from `deploy` function (#4285)
  * REFACTOR-#3642: Move PyArrow storage format usage from main feature to experimental ones (#4374)
  * REFACTOR-#4003: Delete the deprecated cloud mortgage example (#4406)
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
  * FIX-#4422: get rid of case sensitivity for `warns_that_defaulting_to_pandas` (#4423)
  * TEST-#4426: Stop passing is_default kwarg to Modin and pandas (#4428)
  * FIX-#4439: Fix flake8 CI fail (#4440)
  * FIX-#4409: Fix `eval_insert` utility that doesn't actually check results of `insert` function (#4410)
* Documentation improvements
  * DOCS-#4296: Fix docs warnings (#4297)
  * DOCS-#4388: Turn off fail_on_warning option for docs build (#4389)
  * DOCS-#4469: Say that commit messages can start with PERF (#4470).
  * DOCS-#4487: Recommend GitHub issues over feature_requests@modin.org (#4489).
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
@mvashishtha
@anmyachev
@dchigarev
@devin-petersohn
