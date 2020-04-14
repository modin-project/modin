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

engine = os.environ["MODIN_ENGINE"]

header = """<h1 align="center"><img width=7% alt="" src="https://upload.wikimedia.org/wikipedia/commons/thumb/8/86/Teamcity_Logo.png/600px-Teamcity_Logo.png">
    TeamCity {} test results bot</h1>""".format(
    engine.title()
)
if engine == "ray":
    pytest_outputs = ["ray_tests.log"]
elif engine == "dask":
    pytest_outputs = ["dask_tests.log"]
elif engine == "python":
    pytest_outputs = ["python_tests.log"]
else:
    raise Exception("Unknown Engine, set `MODIN_ENGINE` environment variable")

full_comment = header

for out in pytest_outputs:
    full_comment += "\n\n```\n"
    full_comment += open(out, "r").read()
    full_comment += "\n```\n\n</details>\n"

token = os.environ["GITHUB_TOKEN"]
g = Github(token)
repo = g.get_repo("modin-project/modin")

pr = repo.get_pull(pr_id)
if any(
    i.user.login == "modin-bot"
    and "TeamCity {} test results bot".format(engine.title()) in i.body.lower()
    for i in pr.get_issue_comments()
):
    pr_comment_list = [
        i
        for i in list(pr.get_issue_comments())
        if i.user.login == "modin-bot"
        and "TeamCity {} test results bot".format(engine.title()) in i.body.lower()
    ]
    assert len(pr_comment_list) == 1, "Too many comments from modin-bot already"
    pr_comment_list[0].edit(full_comment)
else:
    pr.create_issue_comment(full_comment)
