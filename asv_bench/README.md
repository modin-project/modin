# Modin ASV benchmarks

## Here are some scenarios in which [ASV](https://asv.readthedocs.io/en/stable/index.html) can be used:

* It is necessary to check the impact of the new patch on the performance of a certain set of operations:

  `asv continuous -f 1.05 src/master HEAD -b TimeGroupBy --launch-method=spawn`

* It is necessary to check presence of errors inside of benchmarks after making changes or writing new ones:

  `asv run --quick --show-stderr --python=same --launch-method=spawn`

* You just need to run the entire test suite to get the current time numbers:

  `asv run --launch-method=spawn`

* It is necessary to check the range of commits for performance degradation:

  ```
  asv run [start_hash]..[end_hash] --launch-method=spawn
  asv publish
  asv preview
  ```

For more consistent results, you may need to use the following parameters:

* `-a sample_time=1`
* `-a warmup_time=1`

### Some details about using Modin on Ray with Asv:

* `--launch-method=forkserver` is not working;
* Each set of parameters for each test is launched in its own process, which brings
  a large overhead, since for each process redis server and other necessary binaries
  from ray initialization are started and destroyed.

## How to add a new benchmark?

Basic information on how to write benchmarks - [link](https://asv.readthedocs.io/en/stable/writing_benchmarks.html)

As examples, benchmarks from `benchmarks/benchmarks.py`, `benchmarks/scalability/scalability_benchmarks.py` or `benchmarks/io/csv.py`
can be used.

Requirements:
* the benchmark should be able to run both on modin and on the clean pandas when the appropriate value
of the environment variable `MODIN_ASV_USE_IMPL` is selected.
* the size of the benchmark dataset should depend on the environment variable `MODIN_TEST_DATASET_SIZE`.

## How to change a existing benchmark?

It should be remembered that the hash calculated from the benchmark source code is used to display the results.
When changing the benchmark, the old results will no longer be displayed in the dashboard. In general, this is the correct
behavior so as not to get a situation when incomparable numbers are displayed in the dashboard.
But it should be noted that there are such changes in the source code, in which it is still correct to compare
the "before" and "after" versions, for example, the name of a variable, etc.
In this case, you must either run a new version of the benchmark for all the commits ever counted, or manually change
the hash in the corresponding result files.

## Pipeline for displaying results in a dashboard.

1 step: checking benchmarks for validity, runs in PRs CI.
  During the test, the benchmarks are run once on small data.
  The implementation can be found in `test-asv-benchmarks` job of [ci.yml](https://github.com/modin-project/modin/blob/master/.github/workflows/ci.yml)

2 step: running benchmarks with saving the results in [modin-bench@master](https://github.com/modin-project/modin-bench).
  The launch takes place on its own server, the description of which can be found in the dashboard.
  Run command: `asv run HASHFILE:hashfile.txt --show-stderr --machine xeon-e5 --launch-method=spawn`.
  In the file `hashfile.txt` is the last commit from the master branch.
  The implementation is in the internal teamcity configuration.

3 step: converting the results to html representation, which is saved in [modin-bench@gh-pages](https://github.com/modin-project/modin-bench)
  The implementation can be found in `deploy-gh-pages` job of [push.yml](https://github.com/modin-project/modin-bench/blob/master/.github/workflows/push.yml)
