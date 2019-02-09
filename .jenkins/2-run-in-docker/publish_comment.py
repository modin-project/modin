"""
Post the comment like the following to the PR:
```
🤖 This is a bot message

Test Failed:

- https://s3.amazonaws.com/modin-jenkins-result/123/test_test_1.html
- https://s3.amazonaws.com/modin-jenkins-result/123/test_dataframe.html
```


Expect json schema:
    {"test_name": str, "html": str, "engine", str, "python_version": {str, int}, "partitions": {str, int}}
"""

from github import Github
import os
import argparse
import sys
import glob
import json

parser = argparse.ArgumentParser(description="Pust a comment to PR")
parser.add_argument("--sha", type=str, required=True)
parser.add_argument("--dir", type=str, required=True)
args = parser.parse_args()

failed_json_paths = glob(os.path.join(args.dir, "*.json"))

if len(failed_json_paths) == 0:
    print("No failed test json found, exit now")
    sys.exit(0)

failed_test = [json.load(open(p)) for p in failed_json_paths]

header = """
🤖 Test Failed

Test Failed:
"""

comments = []
for failed in failed_test:
    html = failed.pop("html")
    label = json.dumps(failed)

    comment_line = f"""
    - [{label}](https://s3.amazonaws.com/modin-jenkins-result/{args.sha}/{html}) 
    """.strip("\n")
    comments.append(comment_line)

full_comment = "\n".join(comments)

token = os.environ["GITHUB_TOKEN"]
pr_id = int(os.environ["ghprbPullId"])

g = Github(token)
repo = g.get_repo("modin-project/modin")

pr = repo.get_pull(pr_id)
pr.create_issue_comment(header + full_comment)
