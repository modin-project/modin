FROM python:3.6.6-stretch

COPY requirements/requirements.txt requirements/requirements.txt
RUN pip install -r requirements/requirements.txt

COPY . .
RUN pip install -e .
RUN pip install awscli pytest-html

ENTRYPOINT ["bash", ".jenkins/build-tests/run_test.sh"]
