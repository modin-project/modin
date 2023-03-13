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
  * FIX-#4438: Fix `reindex` function that doesn't preserve initial index metadata (#4442)
  * FIX-#4425: Add parameters to groupby pct_change (#4429)
  * FIX-#4457: Fix `loc` in case when need reindex item (#4457)
  * FIX-#4414: Add missing f prefix on f-strings found at https://codereview.doctor (#4415)
  * FIX-#4461: Fix S3 CSV data path (#4462)
  * FIX-#4467: `drop_duplicates` no longer removes items based on index values (#4468)
  * FIX-#4449: Drain the call queue before waiting on result in benchmark mode (#4472)
  * FIX-#4518: Fix Modin Logging to report specific Modin warnings/errors (#4519)
  * FIX-#4481: Allow clipping with a Modin Series of bounds (#4486)  
  * FIX-#4504: Support na_action in applymap (#4505)
  * FIX-#4503: Stop the memory logging thread after session exit (#4515)
  * FIX-#4531: Fix a makedirs race condition in to_parquet (#4533)
  * FIX-#4464: Refactor Ray utils and quick fix groupby.count failing on virtual partitions (#4490)
  * FIX-#4436: Fix to_pydatetime dtype for timezone None (#4437)
  * FIX-#4541: Fix merge_asof with non-unique right index (#4542)
* Performance enhancements
  * FEAT-#4320: Add connectorx as an alternative engine for read_sql (#4346)
  * PERF-#4493: Use partition size caches more in Modin dataframe (#4495)
* Benchmarking enhancements
  * FEAT-#4371: Add logging to Modin (#4372)
  * FEAT-#4501: Add RSS Memory Profiling to Modin Logging (#4502)
  * FEAT-#4524: Split Modin API and Memory log files (#4526)
* Refactor Codebase
  * REFACTOR-#4284: use variable length unpacking when getting results from `deploy` function (#4285)
  * REFACTOR-#3642: Move PyArrow storage format usage from main feature to experimental ones (#4374)
  * REFACTOR-#4003: Delete the deprecated cloud mortgage example (#4406)
  * REFACTOR-#4513: Fix spelling mistakes in docs and docstrings (#4514)
  * REFACTOR-#4510: Align experimental and regular IO modules initializations (#4511)
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
  * TEST-#4482: Fix getitem and loc with series of bools (#4483).
* Documentation improvements
  * DOCS-#4296: Fix docs warnings (#4297)
  * DOCS-#4388: Turn off fail_on_warning option for docs build (#4389)
  * DOCS-#4469: Say that commit messages can start with PERF (#4470).
  * DOCS-#4466: Recommend GitHub issues over bug_reports@modin.org (#4474).  
  * DOCS-#4487: Recommend GitHub issues over feature_requests@modin.org (#4489).
  * DOCS-#4545: Add socials to README (#4555).
* Dependencies
  * FIX-#4327: Update min pin for xgboost version (#4328)
  * FIX-#4383: Remove `pathlib` from deps (#4384)
  * FIX-#4390: Add `redis` to Modin dependencies (#4396)
  * FIX-#3689: Add black and flake8 into development environment files (#4480)
  * TEST-#4516: Add numpydoc to developer requirements (#4517)
* New Features
  * FEAT-#4412: Add Batch Pipeline API to Modin (#4452)

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
@jrsacher
@orcahmlee
@naren-ponder
@RehanSD
