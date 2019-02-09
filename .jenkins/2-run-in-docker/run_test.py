from subprocess import Popen, check_output
from shlex import split
import sys
import os
import json
from random import random


def exec_cmd(cmd):
    return check_output(split(cmd))


sha_tag = exec_cmd("git rev-parse --verify --short HEAD")
exec_cmd("python .jenkins/utils/strip-type-hints.py")

tests = {
    "dataframe": "modin/pandas/test/test_dataframe.py",
    "concat": "modin/pandas/test/test_concat.py",
    "io": "modin/pandas/test/test_io.py",
    "groupby": "modin/pandas/test/test_groupby.py",
    "io_exp": "modin/experimental/pandas/test/test_io_exp.py",
}
# not using xdist for now
# https://github.com/pytest-dev/pytest-cov/issues/129
par_tests = {}
result_dir = "/result"

engine = os.environ.get("MODIN_ENGINE", "Unknown")
partitions = os.environ.get("MODIN_DEFAULT_NPARTITIONS", "Unknown")
python_ver = sys.version_info[0]

htmls = {
    test_name: "test-{0}_engine-{1}_partition-{2}_python-{3}.html".format(
        test_name, engine, partitions, python_ver
    )
    for test_name in tests.keys()
}

tests_procs = {}
for test_name, test_path in tests.items():
    par_test_flag = "-n auto" if test_name in par_tests else ""

    cmd = """
    pytest {par_test_flag} \
            --html={html} \
            --self-contained-html \
            --disable-pytest-warnings \
            --cov-config=.coveragerc \
            --cov=modin \
            --cov-append \
            {test_path}
    """.strip(
        "\n"
    ).format(
        par_test_flag=par_test_flag, html=htmls[test_name], test_path=test_path
    )

    tests_procs[test_name] = Popen(split(cmd))

failed_procs = []
for test_name, proc in tests_procs.items():
    proc.wait()
    if proc.returncode != 0:
        failed_procs.append(test_name)

        cmd = "aws s3 cp {html} s3://modin-jenkins-result/{sha_tag}/ --acl public-read".format(
            html=htmls[test_name], sha_tag=sha_tag
        )
        exec_cmd(cmd)


if os.path.exists(result_dir) and len(failed_procs) > 0:
    for test_failed in failed_procs:
        with open(os.path.join(result_dir, "{}.json".format(random())), "w") as f:
            json.dump(
                {
                    "test_name": test_failed,
                    "html": htmls[test_failed],
                    "engine": engine,
                    "python_version": python_ver,
                    "partitions": partitions,
                },
                f,
            )

codecov = "curl -s https://codecov.io/bash | bash"
exec_cmd(codecov)
