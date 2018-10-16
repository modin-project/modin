import os

os.makedirs('~/.aws', exists_ok=True)

with open('~/.aws/credentials', 'w') as f:
	f.write(f"""
[default]
aws_access_key_id = {os.environ["AWS_ACCESS_KEY_ID"]}
aws_secret_access_key = {os.environ["AWS_SECRET_ACCESS_KEY"]}
""")
