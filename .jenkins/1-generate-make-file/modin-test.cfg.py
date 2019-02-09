import multiprocessing
num_cpu = multiprocessing.cpu_count()

# assume the following names exists
# from shipyard import Action, CIPrettyLogAction, PRELUDE, IsolatedAction

# Inject sha_tag env var
PRELUDE.append("sha_tag=$(shell git rev-parse --verify --short HEAD)")
PRELUDE.append("pwd=$(shell pwd)")

base_images = {"py2-test": "2.7.15-stretch", "py3-test": "3.6.6-stretch"}
modin_engine_partitions = {"Ray": [4, 8, 12, 16], "Dask": [4], "Python": [4]}


def generate_build_command(image_name, base_image_name):
    return """
    docker build -t modin-project/{image_name} \
            --build-arg PY_VERSION="{base_image_name}" \
            -f .jenkins/Dockerfile .
    """.strip(
        "\n"
    ).format(
        image_name=image_name, base_image_name=base_image_name
    )


def generate_test_command(image_name, engine, partition_size):
    return """
    docker run --rm --shm-size=4g --cpus={num_cpu} \
            -e MODIN_ENGINE={engine} \
            -e MODIN_DEFAULT_NPARTITIONS={partition_size} \
            -e AWS_ACCESS_KEY_ID=$(AWS_ACCESS_KEY_ID) \
            -e AWS_SECRET_ACCESS_KEY=$(AWS_SECRET_ACCESS_KEY) \
            -e GITHUB_TOKEN=$(GITHUB_TOKEN) \
            -e ghprbPullId=$(ghprbPullId) \
            -e CODECOV_TOKEN=$(CODECOV_TOKEN) \
            -v $(pwd)/.jenkins/test-result:/result \
            modin-project/{image_name}
    """.strip(
        "\n"
    ).format(
        image_name=image_name,
        engine=engine,
        partition_size=partition_size,
        num_cpu=min(partition_size * 2, num_cpu),
    )


def generate_report_command():
    return """
    docker run --rm \
            -e GITHUB_TOKEN=$(GITHUB_TOKEN) \
            -e ghprbPullId=$(ghprbPullId) \
            -v $(pwd)/.jenkins/test-result:/result \
            modin-project/py3-test \
            python .jenkins/2-run-in-docker/publish_comment.py \
            --dir /result --sha $(sha_tag)
    """.strip(
        "\n"
    )


# We will use result directory to aggregate test results
result_dir = Action("make_result_dir", command="mkdir .jenkins/test-result")
report_action = IsolatedAction(name="publish_report", command=generate_report_command())

# Here we build the docker images for different python versions
for image_name, base_image in base_images.items():
    CIPrettyLogAction(
        name="build_{}".format(image_name),
        command=generate_build_command(image_name, base_image),
        tags="build",
    )

# Here we test the build matrix according and assign their dependency to
# the docker images they will be run upon
for image_name in base_images.keys():
    for engine, partitions in modin_engine_partitions.items():
        for partition in partitions:
            if image_name == "py2-test" and partition != 4:
                continue

            test_action = CIPrettyLogAction(
                name="test_{0}_{1}_{2}".format(image_name, engine, partition),
                command=generate_test_command(image_name, engine, partition),
                tags="test",
            )

            build_action = Action.get_action("build_{}".format(image_name))
            build_action > test_action
            result_dir > test_action
