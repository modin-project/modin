FROM python:3.6.6-stretch

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . .
RUN pip install -e .
RUN pip install awscli pytest-html

ENTRYPOINT ["bash", ".jenkins/build-tests/run_test.sh"]
