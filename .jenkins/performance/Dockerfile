FROM python:3.6.6-stretch

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
RUN pip install -q pytest pytest-benchmark feather-format lxml openpyxl xlrd numpy matplotlib

COPY . .
RUN pip install -e .

ENTRYPOINT ["bash", ".jenkins/performance/run_perf_test.sh"]
