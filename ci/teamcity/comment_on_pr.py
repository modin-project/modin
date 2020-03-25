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

full_comment = ""
# Do not include coverage info in PR comment
split_by_first = (
    "----------- coverage: platform linux, python 3.7.5-final-0 -----------"
)
split_by_second = "--------------------------------------------------------------------------------------"

for out in pytest_outputs:

    full_comment += "<details><summary>{} Tests</summary>\n".format(
        out.split("_")[0].title()
    )
    full_comment += "\n\n```\n"
    content = open(out, "r").read()
    full_comment += "".join(
        "".join(
            [
                i.split(split_by_first)[0],
                i.split(split_by_first)[-1].split(split_by_second)[-1],
            ]
        )
        for i in content.split("+ python3 -m pytest -n=48 ")
    )
    full_comment += "\n```\n\n</details>\n"

if "FAILURES" not in full_comment:
    header += '<h3 align="center">Tests PASSed</h3>\n\n'
else:
    header += '<h3 align="center">Tests FAILed</h3>\n\n'

full_comment = header + full_comment

token = os.environ["GITHUB_TOKEN"]
g = Github(token)
repo = g.get_repo("modin-project/modin")

pr = repo.get_pull(pr_id)
if len(full_comment) > 65000:
    full_comment = full_comment[-65000:] + "\n\n<b>Remaining output truncated<b>\n\n"
if any(i.user.login == "modin-bot" for i in pr.get_issue_comments()):
    pr_comment_list = [
        i for i in list(pr.get_issue_comments()) if i.user.login == "modin-bot"
    ]
    assert len(pr_comment_list) == 1, "Too many comments from modin-bot already"
    pr_comment_list[0].edit(full_comment)
else:
    pr.create_issue_comment(full_comment)
