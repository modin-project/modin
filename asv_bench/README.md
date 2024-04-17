# Modin ASV benchmarks

## Here are some scenarios in which [ASV](https://asv.readthedocs.io/en/stable/index.html) can be used:

* Check the impact of the new patch on the performance of a certain set of operations:

  `asv continuous -f 1.05 src/main HEAD -b TimeGroupBy --launch-method=spawn`

* Check for presence of errors inside of benchmarks after changing them or writing new ones:

  `asv run --quick --show-stderr --python=same --launch-method=spawn`

* Run entire benchmark suite to get the current times:

  `asv run --launch-method=spawn`

* Check the range of commits for performance degradation:

  ```
  asv run [start_hash]..[end_hash] --launch-method=spawn
  asv publish
  asv preview
  ```

For more consistent results, you may need to use the following parameters which
description is in [ASV docs](https://asv.readthedocs.io/en/stable/benchmarks.html?highlight=sample_time#timing-benchmarks):

* `-a sample_time=1`
* `-a warmup_time=1`

### Notes about using Modin on Ray with Asv:

* `--launch-method=forkserver` is not working;
* Each set of parameters for each test is launched in its own process, which brings
  a large overhead, since for each process redis server and other necessary processes
  from ray initialization are started and destroyed.

## Adding new benchmark

Basic information on writing benchmarks is present [in ASV documentation](https://asv.readthedocs.io/en/stable/writing_benchmarks.html)

Benchmarks from `benchmarks/benchmarks.py`, `benchmarks/scalability/scalability_benchmarks.py` or `benchmarks/io/csv.py`
could be used as a starting point.

Requirements:
* the benchmark should be able to run both on Modin and on Pandas when the appropriate value
of the environment variable `MODIN_ASV_USE_IMPL` is selected.
* the size of the benchmark dataset should depend on the environment variable `MODIN_TEST_DATASET_SIZE`.

## Changing existing benchmark

It should be remembered that the hash calculated from the benchmark source code is used to display the results.
When changing the benchmark, the old results will no longer be displayed in the dashboard. In general, this is the correct
behavior so as not to get a situation when incomparable numbers are displayed in the dashboard.
But it should be noted that there could be changes in the source code when it is still correct to compare
the "before" and "after" versions, for example, name of a variable changed, comment added, etc.
In this case you must either run a new version of the benchmark for all the commits ever accounted for or manually change
the hash in the corresponding result files.

## Pipeline for displaying results in a dashboard

Step 1: checking benchmarks for validity, runs in PRs CI.
  During the test, the benchmarks are run once on small data.
  The implementation can be found in `test-asv-benchmarks` job of [ci.yml](https://github.com/modin-project/modin/blob/main/.github/workflows/ci.yml)

Step 2: running benchmarks with saving the results in [modin-bench@master](https://github.com/modin-project/modin-bench).
  The launch takes place on internal server using specific TeamCity configuration.
  The description of the server can be found in the ["Benchmark list"](https://modin.org/modin-bench/#summarylist?sort=0&dir=asc) tab,
  on the left when you hover the mouse over the machine name. 
  This step starts as scheduled (now every half hour), subject to the presence of new commits in the Modin `main` branch.
  Command to run benchmarks: `asv run HASHFILE:hashfile.txt --show-stderr --machine xeon-e5 --launch-method=spawn`.
  In the file `hashfile.txt` is the last modin commit hash.
  Writing to a `modin-bench@master` triggers 3 step of the pipeline.

Step 3: converting the results to html representation, which is saved in [modin-bench@gh-pages](https://github.com/modin-project/modin-bench)
  The implementation can be found in `deploy-gh-pages` job of [push.yml](https://github.com/modin-project/modin-bench/blob/master/.github/workflows/push.yml)

Basic actions for step 2:
* setup environment variable:
  * export MODIN_TEST_DATASET=Big
  * export MODIN_CPUS=44
* setup git client
* prepare json file with machine description
  * This file should be placed in the user's home directory.
  * ASV does not always automatically create the file with the description of the machine correctly (e.g. due to being run in a container).
  It is recommended to create a file using [asv machine](https://asv.readthedocs.io/en/stable/commands.html?highlight=machine%20description#asv-machine) command, and manually check the result.
  [Example](https://github.com/modin-project/modin-bench/blob/master/results/xeon-e5/machine.json)
* copy old result to folder where new result will appear
  (conflict resolution will be performed by ASV itself instead of git)
* push performance result to modin-bench repository
