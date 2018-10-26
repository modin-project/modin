set -ex

sha_tag=`git rev-parse --verify --short HEAD`
docker build -t modin-project/perf-test:$sha_tag -f .jenkins/performance-tests/Dockerfile .
docker run --rm --shm-size=1g \
	-e AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID} \
	-e AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY} \
	modin-project/perf-test:$sha_tag
