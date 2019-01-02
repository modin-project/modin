"""
Post the comment like the following to the PR:
```
🤖 This is a bot message

Test Failed:

- https://s3.amazonaws.com/modin-jenkins-result/123/test_test_1.html
- https://s3.amazonaws.com/modin-jenkins-result/123/test_dataframe.html
```

Parameterized by passing in --tests and --sha
"""

from github import Github
import os
import argparse
import sys

parser = argparse.ArgumentParser(description="Pust a comment to PR")
parser.add_argument("--tests", type=str, required=True)
parser.add_argument("--sha", type=str, required=True)
args = parser.parse_args()

tests_ran = args.tests.split(' ')

if len(tests_ran) == 0:
    sys.exit(0)

header = """
🤖 This is a bot message

Test Failed:
"""
comment = '\n'.join([
    f"- https://s3.amazonaws.com/modin-jenkins-result/{args.sha}/test_{test_name}.html"
    for test_name in tests_ran
])

token = os.environ["GITHUB_TOKEN"]
pr_id = int(os.environ["ghprbPullId"])

g = Github(token)
repo = g.get_repo("modin-project/modin")

pr = repo.get_pull(pr_id)
pr.create_issue_comment(header+comment)
