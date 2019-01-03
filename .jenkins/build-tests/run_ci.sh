set -e
set -x

echo "Check something, will remove soon - Simon"
git log | head -n 100

echo ${ghprbActualCommit}

exit 1

sha_tag=`git rev-parse --verify --short HEAD`
docker build -t modin-project/test:$sha_tag -f .jenkins/build-tests/Dockerfile .
docker run --rm --shm-size=1g \
	-e AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID} \
	-e AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY} \
	-e GITHUB_TOKEN=${GITHUB_TOKEN} \
	-e ghprbPullId=${ghprbPullId} \
	modin-project/test:$sha_tag
