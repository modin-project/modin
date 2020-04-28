FROM python:3.6.6-stretch

COPY env/requirements.txt requirements.txt
RUN pip install -r env/requirements.txt
RUN pip install -q pytest==3.9.3 awscli pytest-benchmark feather-format lxml openpyxl xlrd numpy matplotlib sqlalchemy

COPY . .
RUN pip install -e .

ENTRYPOINT ["bash", ".jenkins/performance-tests/run_perf_test.sh"]
