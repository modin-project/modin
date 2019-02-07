set -e
set -x

sha_tag=`git rev-parse --verify --short HEAD`
docker build -t modin-project/test:$sha_tag -f .jenkins/build-tests/Dockerfile .
docker run --rm --shm-size=16g --cpus=4 \
    -e MODIN_DEBUG=1 \
	-e AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID} \
	-e AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY} \
	-e GITHUB_TOKEN=${GITHUB_TOKEN} \
	-e ghprbPullId=${ghprbPullId} \
	modin-project/test:$sha_tag
