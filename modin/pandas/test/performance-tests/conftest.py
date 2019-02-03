# import subprocess
# import os
#
#
# def pytest_benchmark_update_commit_info(config, commit_info):
#     commit_info["commit_number"] = int(
#         subprocess.check_output(["git", "rev-list", "HEAD", "--count"]).strip()
#     )
#     commit_info["engine"] = os.environ.get("MODIN_ENGINE", "Ray").title()
