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
  * FIX-#4914: `base_lengths` should be computed from `base_frame` instead of `self` in `copartition` (#4915)
  * FIX-#4848: Fix rebalancing partitions when NPartitions == 1 (#4874)
  * FIX-#4927: Fix `dtypes` computation in `dataframe.filter` (#4928)
  * FIX-#4907: Implement `radd` for Series and DataFrame (#4908)
  * FIZ-#4945: Fix `_take_2d_positional` that loses indexes due to filtering empty dataframes (#4951)
  * FIX-#4818, PERF-#4825: Fix where by using the new n-ary operator (#4820)
  * FIX-#3983: FIX-#4107: Materialize 'rowid' columns when selecting rows by position (#4834)
  * FIX-#4845: Fix KeyError from `__getitem_bool` for single row dataframes (#4845)
  * FIX-#4734: Handle Series.apply when return type is a DataFrame (#4830)
  * FIX-#4983: Set `frac` to `None` in _sample when `n=0` (#4984)
  * FIX-#4993: Return `_default_to_pandas` in `df.attrs` (#4995)
  * FIX-#5043: Fix `execute` function in ASV utils failed if `len(partitions) == 0` (#5044)
  * FIX-#4597: Refactor Partition handling of func, args, kwargs (#4715)
  * FIX-#4996: Evaluate BenchmarkMode at each function call (#4997)
  * FIX-#4022: Fixed empty data frame with index (#4910)
  * FIX-#4090: Fixed check if the index is trivial (#4936)
  * FIX-#4966: Fix `to_timedelta` to return Series instead of TimedeltaIndex (#5028)
  * FIX-#5042: Fix series __getitem__ with invalid strings (#5048)
  * FIX-#4691: Fix binary operations between virtual partitions (#5049)  
  * FIX-#5045: Fix ray virtual_partition.wait with duplicate object refs (#5058)
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
  * PERF-#4849: Compute `dtypes` in `concat` also for ROW_WISE case when possible (#4850)
  * PERF-#4929: Compute `dtype` when using `Series.dt` accessor (#4930)
  * PERF-#4892: Compute `lengths` in `rebalance_partitions` when possible (#4893)
  * PERF-#4794: Compute caches in `_propagate_index_objs` (#4888)
  * PERF-#4860: `PandasDataframeAxisPartition.deploy_axis_func` should be serialized only once (#4861)
  * PERF-#4890: `PandasDataframeAxisPartition.drain` should be serialized only once (#4891)
  * PERF-#4870: Avoid index materialization in `__getattribute__` and `__getitem__` (4911)
  * PERF-#4886: Use lazy index and columns evaluation in `query` method (#4887)
  * PERF-#4866: `iloc` function that used in `partition.mask` should be serialized only once (#4901)
  * PERF-#4920: Avoid index and cache computations in `take_2d_labels_or_positional` unless they are needed (#4921)
  * PERF-#4999: don't call `apply` in virtual partition' `drain_call_queue` if `call_queue` is empty (#4975)
  * PERF-#4268: Implement partition-parallel __getitem__ for bool Series masks (#4753)
  * PERF-#5017: `reset_index` shouldn't trigger index materialization if possible (#5018)
  * PERF-#4963: Use partition `width/length` methods instead of `_compute_axis_labels_and_lengths` if index is already known (#4964)
  * PERF-#4940: Optimize categorical dtype check in `concatenate` (#4953)
* Benchmarking enhancements
  * TEST-#5066: Add outer join case for `TimeConcat` benchmark (#5067)
  * TEST-#5083: Add `merge` op with categorical data (#5084)
  * FEAT-#4706: Add Modin ClassLogger to PandasDataframePartitionManager (#4707)
  * TEST-#5014: Simplify adding new ASV benchmarks (#5015)
  * TEST-#5064: Update `TimeConcat` benchmark with new parameter `ignore_index` (#5065)
  * PERF-#4944: Avoid default_to_pandas in ``Series.cat.codes``, ``Series.dt.tz``, and ``Series.dt.to_pytimedelta`` (#4833)
  * TEST-#5068: Add binary op benchmark for Series (#5069)
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
  * REFACTOR-#4000: Remove code duplication for `PandasOnRayDataframePartitionManager` (#4895)
  * REFACTOR-#3780: Remove code duplication for `PandasOnDaskDataframe` (#3781)
  * REFACTOR-#4530: Unify access to physical data for any partition type (#4829)
  * REFACTOR-#4978: Align `modin/core/execution/dask/common/__init__.py` with `modin/core/execution/ray/common/__init__.py` (#4979)
  * REFACTOR-#4949: Remove code duplication in `default2pandas/dataframe.py` and `default2pandas/any.py` (#4950)
  * REFACTOR-#4976: Rename `RayTask` to `RayWrapper` in accordance with Dask (#4977)
  * REFACTOR-#4885: De-duplicated take_2d_labels_or_positional methods (#4883)
  * REFACTOR-#5005: Use `finalize` method instead of list comprehension + `drain_call_queue` (#5006)
  * REFACTOR-#5001: Remove `jenkins` stuff (#5002)
  * REFACTOR-#5026: Change exception names to simplify grepping (#5027)
  * REFACTOR-#4970: Rewrite base implementations of a partition' `width/length` (#4971)  
  * REFACTOR-#4942: Remove `call` method in favor of `register` due to duplication (4943)
  * REFACTOR-#4922: Helpers for take_2d_labels_or_positional (#4865)
  * REFACTOR-#5024: Make `_row_lengths` and `_column_widths` public (#5025)
  * REFACTOR-#5009: Use `RayWrapper.materialize` instead of `ray.get` (#5010)
  * REFACTOR-#4755: Rewrite Pandas version mismatch warning (#4965)
  * REFACTOR-#5012: Add mypy checks for singleton files in base modin directory (#5013)
  * REFACTOR-#5038: Remove unnecessary _method argument from resamplers (#5039)
  * REFACTOR-#5081: Remove `c323f7fe385011ed849300155de07645.db` file (#5082)
* Pandas API implementations and improvements
  * FEAT-#4670: Implement convert_dtypes by mapping across partitions (#4671)
* OmniSci enhancements
  * FEAT-#4913: Enabling pyhdk
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
  * TEST-#4879: Use pandas `ensure_clean()` in place of `io_tests_data` (#4881)
  * TEST-#4562: Use local Ray cluster in CI to resolve flaky `test-compat-win` (#5007)
  * TEST-#5040: Rework test_series using eval_general() (#5041)
  * TEST-#5050: Add black to pre-commit hook (#5051)
* Documentation improvements
  * DOCS-#4552: Change default sphinx language to en to fix sphinx >= 5.0.0 build (#4553)
  * DOCS-#4628: Add to_parquet partial support notes (#4648)
  * DOCS-#4668: Set light theme for readthedocs page, remove theme switcher (#4669)
  * DOCS-#4748: Apply the Triage label to new issues (#4749)
  * DOCS-#4790: Give all templates issue type and triage labels (#4791)
  * DOCS-#4521: Document how to benchmark modin (#5020)
* Dependencies
  * FEAT-#4598: Add support for pandas 1.4.3 (#4599)
  * FEAT-#4619: Integrate mypy static type checking (#4620)
  * FEAT-#4202: Allow dask past 2022.2.0 (#4769)
  * FEAT-#4925: Upgrade pandas to 1.4.4 (#4926)
  * TEST-#4998: Add flake8 plugins to dev requirements (#5000)
* New Features
  * FEAT-4463: Add experimental fuzzydata integration for testing against a randomized dataframe workflow (#4556)
  * FEAT-#4419: Extend virtual partitioning API to pandas on Dask (#4420)
  * FEAT-#4147: Add partial compatibility with Python 3.6 and pandas 1.1 (#4301)
  * FEAT-#4569: Add error message when `read_` function defaults to pandas (#4647)
  * FEAT-#4725: Make index and columns lazy in Modin DataFrame (#4726)
  * FEAT-#4664: Finalize compatibility support for Python 3.6 (#4800)
  * FEAT-#4746: Sync interchange protocol with recent API changes (#4763)
  * FEAT-#4733: Support fastparquet as engine for `read_parquet` (#4807)
  * FEAT-#4766: Support fsspec URLs in `read_csv` and `read_csv_glob` (#4898)
  * FEAT-#4827: Implement `infer_types` dataframe algebra operator (#4871)
  * FEAT-#4989: Switch pandas version to 1.5 (#5037)

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
@ienkovich
@Garra1980
@Billy2551
