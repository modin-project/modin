import subprocess


def pytest_benchmark_update_commit_info(config, commit_info):
    commit_info["commit_number"] = int(
        subprocess.check_output(["git", "rev-list", "HEAD", "--count"]).strip()
    )
