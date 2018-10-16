set -ex

sha_tag=`git rev-parse --verify --short HEAD`
docker build -t modin-project/test:$sha_tag -f .jenkins/Dockerfile .
docker run --rm --shm-size=1g --cpuset-cpus="0-3" modin-project/test:$sha_tag

