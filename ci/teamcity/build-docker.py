import os
import sys


def execute_command(cmd):
    status = os.system(cmd)
    ec = os.WEXITSTATUS(status)
    if ec != 0:
        raise SystemExit('Command "{}" failed'.format(cmd))


if sys.platform.startswith("linux"):
    execute_command("git rev-parse HEAD > git-rev")
    execute_command(
        "(cd ../.. && git archive -o ci/teamcity/modin.tar $(cat ci/teamcity/git-rev))"
    )
    base_image = "ray-project/deploy"
    requirements = "requirements-dev.txt"
    execute_command(
        "docker build -f Dockerfile.modin-base --build-arg BASE_IMAGE={} -t modin-project/modin-base .".format(
            base_image
        )
    )
else:
    raise SystemExit(
        "TeamCity CI in Docker containers is supported only on Linux at the moment."
    )

execute_command(
    "docker build -f Dockerfile.teamcity-ci --build-arg REQUIREMENTS={} -t modin-project/teamcity-ci .".format(
        requirements
    )
)

if sys.platform.startswith("linux"):
    execute_command("rm ./modin.tar ./git-rev")
