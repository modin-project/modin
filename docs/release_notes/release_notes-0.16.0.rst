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
  * FIX-#4680: Fix `read_csv` that started defaulting to pandas again in case of reading from a buffer and when a buffer has a non-zero starting position (#4681)
  * FIX-#4491: Wait for all partitions in parallel in benchmark mode (#4656)
  * FIX-#4358: MultiIndex `loc` shouldn't drop levels for full-key lookups (#4608)
  * FIX-#4658: Expand exception handling for `read_*` functions from s3 storages (#4659)
  * FIX-#4672: Fix incorrect warning when setting `frame.index` or `frame.columns` (#4721)
  * FIX-#4686: Propagate metadata and drain call queue in unwrap_partitions (#4697)
  * FIX-#4652: Support categorical data in `from_dataframe` (#4737)
  * FIX-#4756: Correctly propagate `storage_options` in `read_parquet` (#4764)
  * FIX-#4657: Use `fsspec` for handling s3/http-like paths instead of `s3fs` (#4710)
  * FIX-#4676: drain sub-virtual-partition call queues (#4695)
  * FIX-#4782: Exclude certain non-parquet files in `read_parquet` (#4783)    
  * FIX-#4808: Set dtypes correctly after column rename (#4809)
  * FIX-#4811: Apply dataframe -> not_dataframe functions to virtual partitions (#4812)
  * FIX-#4099: Use mangled column names but keep the original when building frames from arrow (#4767)
  * FIX-#4838: Bump up modin-spreadsheet to latest master (#4839)
  * FIX-#4840: Change modin-spreadsheet version for notebook requirements (#4841)
  * FIX-#4835: Handle Pathlike paths in `read_parquet` (#4837)
  * FIX-#4872: Stop checking the private ray mac memory limit (#4873)
  * FIX-#4848: Fix rebalancing partitions when NPartitions == 1 (#4874)
* Performance enhancements
  * PERF-#4182: Add cell-wise execution for binary ops, fix bin ops for empty dataframes (#4391)
  * PERF-#4288: Improve perf of `groupby.mean` for narrow data (#4591)
  * PERF-#4772: Remove `df.copy` call from `from_pandas` since it is not needed for Ray and Dask (#4781)
  * PERF-#4325: Improve perf of multi-column assignment in `__setitem__` when no new column names are assigning (#4455)
  * PERF-#3844: Improve perf of `drop` operation (#4694)
  * PERF-#4727: Improve perf of `concat` operation (#4728)
  * PERF-#4705: Improve perf of arithmetic operations between `Series` objects with shared `.index` (#4689)
  * PERF-#4703: Improve performance in accessing `ser.cat.categories`, `ser.cat.ordered`, and `ser.__array_priority__` (#4704)
  * PERF-#4305: Parallelize `read_parquet` over row groups (#4700)
  * PERF-#4773: Compute `lengths` and `widths` in `put` method of Dask partition like Ray do (#4780)
  * PERF-#4732: Avoid overwriting already-evaluated `PandasOnRayDataframePartition._length_cache` and `PandasOnRayDataframePartition._width_cache` (#4754)
  * PERF-#4862: Don't call `compute_sliced_len.remote` when `row_labels/col_labels == slice(None)` (#4863)
  * PERF-#4713: Stop overriding the ray MacOS object store size limit (#4792)
  * PERF-#4851: Compute `dtypes` for binary operations that can only return bool type and the right operand is not a Modin object (#4852)
  * PERF-#4842: `copy` should not trigger any previous computations (#4843)
  * PERF-#4849: compute `dtypes` in `concat` also for ROW_WISE case when possible (#4850)
  * PERF-#4794: Compute caches in `_propagate_index_objs` (#4888)
  * PERF-#4860: `PandasDataframeAxisPartition.deploy_axis_func` should be serialized only once (#4861)
  * PERF-#4268: Implement partition-parallel __getitem__ for bool Series masks (#4753)
* Benchmarking enhancements
  * FEAT-#4706: Add Modin ClassLogger to PandasDataframePartitionManager (#4707)
* Refactor Codebase
  * REFACTOR-#4530: Standardize access to physical data in partitions (#4563)
  * REFACTOR-#4534: Replace logging meta class with class decorator (#4535)
  * REFACTOR-#4708: Delete combine dtypes (#4709)
  * REFACTOR-#4629: Add type annotations to modin/config (#4685)
  * REFACTOR-#4717: Improve PartitionMgr.get_indices() usage (#4718)
  * REFACTOR-#4730: make Indexer immutable (#4731)
  * REFACTOR-#4774: remove `_build_treereduce_func` call from `_compute_dtypes` (#4775)
  * REFACTOR-#4750: Delete BaseDataframeAxisPartition.shuffle (#4751)
  * REFACTOR-#4722: Stop suppressing undefined name lint (#4723)
  * REFACTOR-#4832: unify `split_result_of_axis_func_pandas` (#4831)
  * REFACTOR-#4796: Introduce constant for __reduced__ column name (#4799)
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
  * TEST-#4698: Stop passing invalid storage_options param (#4699)
  * TEST-#4745: Pin flake8 to <5 to workaround installation conflict (#4752)
  * TEST-#4875: XFail tests failing due to file gone missing (#4876)
* Documentation improvements
  * DOCS-#4552: Change default sphinx language to en to fix sphinx >= 5.0.0 build (#4553)
  * DOCS-#4628: Add to_parquet partial support notes (#4648)
  * DOCS-#4668: Set light theme for readthedocs page, remove theme switcher (#4669)
  * DOCS-#4748: Apply the Triage label to new issues (#4749)
  * DOCS-#4790: Give all templates issue type and triage labels (#4791)
* Dependencies
  * FEAT-#4598: Add support for pandas 1.4.3 (#4599)
  * FEAT-#4619: Integrate mypy static type checking (#4620)
  * FEAT-#4202: Allow dask past 2022.2.0 (#4769)
* New Features
  * FEAT-4463: Add experimental fuzzydata integration for testing against a randomized dataframe workflow (#4556)
  * FEAT-#4419: Extend virtual partitioning API to pandas on Dask (#4420)
  * FEAT-#4147: Add partial compatibility with Python 3.6 and pandas 1.1 (#4301)
  * FEAT-#4569: Add error message when `read_` function defaults to pandas (#4647)
  * FEAT-#4725: Make index and columns lazy in Modin DataFrame (#4726)
  * FEAT-#4664: Finalize compatibility support for Python 3.6 (#4800)
  * FEAT-#4746: Sync interchange protocol with recent API changes (#4763)
  * FEAT-#4733: Support fastparquet as engine for `read_parquet` (#4807)

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
@naren-ponder
@jbrockmendel
