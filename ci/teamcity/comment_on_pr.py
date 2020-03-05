"""
Post the comment like the following to the PR:
```
:robot: TeamCity test results bot :robot:

<Logs from pytest>
```
"""

from github import Github
import os
import sys

# Check if this is a pull request or not based on the environment variable
try:
    pr_id = int(os.environ["GITHUB_PR_NUMBER"].split("/")[-1])
except Exception:
    sys.exit(0)
header = """
<h2 align="center">:robot: TeamCity test results bot :robot:</h2>

"""

pytest_outputs = ["ray_tests.log", "dask_tests.log", "python_tests.log"]

full_comment = header

for out in pytest_outputs:

    full_comment += "<details><summary>{} Tests</summary>\n".format(
        out.split("_")[0].title()
    )
    full_comment += "\n\n```\n"
    full_comment += open(out, "r").read()
    full_comment += "\n```\n\n</details>\n"

token = os.environ["GITHUB_TOKEN"]
g = Github(token)
repo = g.get_repo("modin-project/modin")

pr = repo.get_pull(pr_id)
if any(i.user.login == "modin-bot" for i in pr.get_issue_comments()):
    pr_comment_list = [
        i for i in list(pr.get_issue_comments()) if i.user.login == "modin-bot"
    ]
    assert len(pr_comment_list) == 1, "Too many comments from modin-bot already"
    pr_comment_list[0].edit(full_comment)
else:
    pr.create_issue_comment(full_comment)
