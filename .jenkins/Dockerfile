ARG PY_VERSION
FROM python:${PY_VERSION}

COPY requirements.txt requirements.txt
RUN pip -qq install -r requirements.txt

COPY . .
RUN pip -qq install -e .
RUN pip -qq install awscli pytest-html PyGithub

CMD ["python", ".jenkins/2-run-in-docker/run_test.py"]
