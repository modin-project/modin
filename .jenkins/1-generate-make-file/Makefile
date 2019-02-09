SHELL=/bin/bash -o pipefail
sha_tag=$(shell git rev-parse --verify --short HEAD)

placeholder: 
	echo "Do not run make without any target!"
        

make_result_dir: 
	mkdir .jenkins/test-result
        

publish_report: test_py2-test_Ray_4 test_py2-test_Ray_8 test_py2-test_Ray_12 test_py2-test_Ray_16 test_py2-test_Dask_4 test_py2-test_Python_4 test_py3-test_Ray_4 test_py3-test_Ray_8 test_py3-test_Ray_12 test_py3-test_Ray_16 test_py3-test_Dask_4 test_py3-test_Python_4
		@echo ===== start: publish_report =====
	
		(    docker run --rm             -e MODIN_ENGINE={engine}             -e MODIN_DEFAULT_NPARTITIONS={partition_size}             -e AWS_ACCESS_KEY_ID=$(AWS_ACCESS_KEY_ID)             -e AWS_SECRET_ACCESS_KEY=$(AWS_SECRET_ACCESS_KEY)             -e GITHUB_TOKEN=$(GITHUB_TOKEN)             -e ghprbPullId=$(ghprbPullId)             -v .jenkins/test-result:/result             modin-project/{image_name}:$(sha_tag)             bash .jenkins/publish-report.sh) 2>&1 | python3 ./jenkins/utils/colorize_output.py --tag publish_report
	
		@echo ===== finished: publish_report =====
	
        

build_py2-test: 
		@echo ===== start: build_py2-test =====
	
		(    docker build -t modin-project/py2-test             --build-arg PY_VERSION="2.7.15-stretch             -f .jenkins/build-tests/Dockerfile .) 2>&1 | python3 ./jenkins/utils/colorize_output.py --tag build_py2-test
	
		@echo ===== finished: build_py2-test =====
	
        

build_py3-test: 
		@echo ===== start: build_py3-test =====
	
		(    docker build -t modin-project/py3-test             --build-arg PY_VERSION="3.6.6-stretch             -f .jenkins/build-tests/Dockerfile .) 2>&1 | python3 ./jenkins/utils/colorize_output.py --tag build_py3-test
	
		@echo ===== finished: build_py3-test =====
	
        

test_py2-test_Ray_4: build_py2-test make_result_dir
		@echo ===== start: test_py2-test_Ray_4 =====
	
		(    docker run --rm --shm-size=4g --cpus=4             -e MODIN_ENGINE=Ray             -e MODIN_DEFAULT_NPARTITIONS=4             -e AWS_ACCESS_KEY_ID=$(AWS_ACCESS_KEY_ID)             -e AWS_SECRET_ACCESS_KEY=$(AWS_SECRET_ACCESS_KEY)             -e GITHUB_TOKEN=$(GITHUB_TOKEN)             -e ghprbPullId=$(ghprbPullId)             -v .jenkins/test-result:/result             modin-project/py2-test:$(sha_tag)) 2>&1 | python3 ./jenkins/utils/colorize_output.py --tag test_py2-test_Ray_4
	
		@echo ===== finished: test_py2-test_Ray_4 =====
	
        

test_py2-test_Ray_8: build_py2-test make_result_dir
		@echo ===== start: test_py2-test_Ray_8 =====
	
		(    docker run --rm --shm-size=4g --cpus=4             -e MODIN_ENGINE=Ray             -e MODIN_DEFAULT_NPARTITIONS=8             -e AWS_ACCESS_KEY_ID=$(AWS_ACCESS_KEY_ID)             -e AWS_SECRET_ACCESS_KEY=$(AWS_SECRET_ACCESS_KEY)             -e GITHUB_TOKEN=$(GITHUB_TOKEN)             -e ghprbPullId=$(ghprbPullId)             -v .jenkins/test-result:/result             modin-project/py2-test:$(sha_tag)) 2>&1 | python3 ./jenkins/utils/colorize_output.py --tag test_py2-test_Ray_8
	
		@echo ===== finished: test_py2-test_Ray_8 =====
	
        

test_py2-test_Ray_12: build_py2-test make_result_dir
		@echo ===== start: test_py2-test_Ray_12 =====
	
		(    docker run --rm --shm-size=4g --cpus=4             -e MODIN_ENGINE=Ray             -e MODIN_DEFAULT_NPARTITIONS=12             -e AWS_ACCESS_KEY_ID=$(AWS_ACCESS_KEY_ID)             -e AWS_SECRET_ACCESS_KEY=$(AWS_SECRET_ACCESS_KEY)             -e GITHUB_TOKEN=$(GITHUB_TOKEN)             -e ghprbPullId=$(ghprbPullId)             -v .jenkins/test-result:/result             modin-project/py2-test:$(sha_tag)) 2>&1 | python3 ./jenkins/utils/colorize_output.py --tag test_py2-test_Ray_12
	
		@echo ===== finished: test_py2-test_Ray_12 =====
	
        

test_py2-test_Ray_16: build_py2-test make_result_dir
		@echo ===== start: test_py2-test_Ray_16 =====
	
		(    docker run --rm --shm-size=4g --cpus=4             -e MODIN_ENGINE=Ray             -e MODIN_DEFAULT_NPARTITIONS=16             -e AWS_ACCESS_KEY_ID=$(AWS_ACCESS_KEY_ID)             -e AWS_SECRET_ACCESS_KEY=$(AWS_SECRET_ACCESS_KEY)             -e GITHUB_TOKEN=$(GITHUB_TOKEN)             -e ghprbPullId=$(ghprbPullId)             -v .jenkins/test-result:/result             modin-project/py2-test:$(sha_tag)) 2>&1 | python3 ./jenkins/utils/colorize_output.py --tag test_py2-test_Ray_16
	
		@echo ===== finished: test_py2-test_Ray_16 =====
	
        

test_py2-test_Dask_4: build_py2-test make_result_dir
		@echo ===== start: test_py2-test_Dask_4 =====
	
		(    docker run --rm --shm-size=4g --cpus=4             -e MODIN_ENGINE=Dask             -e MODIN_DEFAULT_NPARTITIONS=4             -e AWS_ACCESS_KEY_ID=$(AWS_ACCESS_KEY_ID)             -e AWS_SECRET_ACCESS_KEY=$(AWS_SECRET_ACCESS_KEY)             -e GITHUB_TOKEN=$(GITHUB_TOKEN)             -e ghprbPullId=$(ghprbPullId)             -v .jenkins/test-result:/result             modin-project/py2-test:$(sha_tag)) 2>&1 | python3 ./jenkins/utils/colorize_output.py --tag test_py2-test_Dask_4
	
		@echo ===== finished: test_py2-test_Dask_4 =====
	
        

test_py2-test_Python_4: build_py2-test make_result_dir
		@echo ===== start: test_py2-test_Python_4 =====
	
		(    docker run --rm --shm-size=4g --cpus=4             -e MODIN_ENGINE=Python             -e MODIN_DEFAULT_NPARTITIONS=4             -e AWS_ACCESS_KEY_ID=$(AWS_ACCESS_KEY_ID)             -e AWS_SECRET_ACCESS_KEY=$(AWS_SECRET_ACCESS_KEY)             -e GITHUB_TOKEN=$(GITHUB_TOKEN)             -e ghprbPullId=$(ghprbPullId)             -v .jenkins/test-result:/result             modin-project/py2-test:$(sha_tag)) 2>&1 | python3 ./jenkins/utils/colorize_output.py --tag test_py2-test_Python_4
	
		@echo ===== finished: test_py2-test_Python_4 =====
	
        

test_py3-test_Ray_4: build_py3-test make_result_dir
		@echo ===== start: test_py3-test_Ray_4 =====
	
		(    docker run --rm --shm-size=4g --cpus=4             -e MODIN_ENGINE=Ray             -e MODIN_DEFAULT_NPARTITIONS=4             -e AWS_ACCESS_KEY_ID=$(AWS_ACCESS_KEY_ID)             -e AWS_SECRET_ACCESS_KEY=$(AWS_SECRET_ACCESS_KEY)             -e GITHUB_TOKEN=$(GITHUB_TOKEN)             -e ghprbPullId=$(ghprbPullId)             -v .jenkins/test-result:/result             modin-project/py3-test:$(sha_tag)) 2>&1 | python3 ./jenkins/utils/colorize_output.py --tag test_py3-test_Ray_4
	
		@echo ===== finished: test_py3-test_Ray_4 =====
	
        

test_py3-test_Ray_8: build_py3-test make_result_dir
		@echo ===== start: test_py3-test_Ray_8 =====
	
		(    docker run --rm --shm-size=4g --cpus=4             -e MODIN_ENGINE=Ray             -e MODIN_DEFAULT_NPARTITIONS=8             -e AWS_ACCESS_KEY_ID=$(AWS_ACCESS_KEY_ID)             -e AWS_SECRET_ACCESS_KEY=$(AWS_SECRET_ACCESS_KEY)             -e GITHUB_TOKEN=$(GITHUB_TOKEN)             -e ghprbPullId=$(ghprbPullId)             -v .jenkins/test-result:/result             modin-project/py3-test:$(sha_tag)) 2>&1 | python3 ./jenkins/utils/colorize_output.py --tag test_py3-test_Ray_8
	
		@echo ===== finished: test_py3-test_Ray_8 =====
	
        

test_py3-test_Ray_12: build_py3-test make_result_dir
		@echo ===== start: test_py3-test_Ray_12 =====
	
		(    docker run --rm --shm-size=4g --cpus=4             -e MODIN_ENGINE=Ray             -e MODIN_DEFAULT_NPARTITIONS=12             -e AWS_ACCESS_KEY_ID=$(AWS_ACCESS_KEY_ID)             -e AWS_SECRET_ACCESS_KEY=$(AWS_SECRET_ACCESS_KEY)             -e GITHUB_TOKEN=$(GITHUB_TOKEN)             -e ghprbPullId=$(ghprbPullId)             -v .jenkins/test-result:/result             modin-project/py3-test:$(sha_tag)) 2>&1 | python3 ./jenkins/utils/colorize_output.py --tag test_py3-test_Ray_12
	
		@echo ===== finished: test_py3-test_Ray_12 =====
	
        

test_py3-test_Ray_16: build_py3-test make_result_dir
		@echo ===== start: test_py3-test_Ray_16 =====
	
		(    docker run --rm --shm-size=4g --cpus=4             -e MODIN_ENGINE=Ray             -e MODIN_DEFAULT_NPARTITIONS=16             -e AWS_ACCESS_KEY_ID=$(AWS_ACCESS_KEY_ID)             -e AWS_SECRET_ACCESS_KEY=$(AWS_SECRET_ACCESS_KEY)             -e GITHUB_TOKEN=$(GITHUB_TOKEN)             -e ghprbPullId=$(ghprbPullId)             -v .jenkins/test-result:/result             modin-project/py3-test:$(sha_tag)) 2>&1 | python3 ./jenkins/utils/colorize_output.py --tag test_py3-test_Ray_16
	
		@echo ===== finished: test_py3-test_Ray_16 =====
	
        

test_py3-test_Dask_4: build_py3-test make_result_dir
		@echo ===== start: test_py3-test_Dask_4 =====
	
		(    docker run --rm --shm-size=4g --cpus=4             -e MODIN_ENGINE=Dask             -e MODIN_DEFAULT_NPARTITIONS=4             -e AWS_ACCESS_KEY_ID=$(AWS_ACCESS_KEY_ID)             -e AWS_SECRET_ACCESS_KEY=$(AWS_SECRET_ACCESS_KEY)             -e GITHUB_TOKEN=$(GITHUB_TOKEN)             -e ghprbPullId=$(ghprbPullId)             -v .jenkins/test-result:/result             modin-project/py3-test:$(sha_tag)) 2>&1 | python3 ./jenkins/utils/colorize_output.py --tag test_py3-test_Dask_4
	
		@echo ===== finished: test_py3-test_Dask_4 =====
	
        

test_py3-test_Python_4: build_py3-test make_result_dir
		@echo ===== start: test_py3-test_Python_4 =====
	
		(    docker run --rm --shm-size=4g --cpus=4             -e MODIN_ENGINE=Python             -e MODIN_DEFAULT_NPARTITIONS=4             -e AWS_ACCESS_KEY_ID=$(AWS_ACCESS_KEY_ID)             -e AWS_SECRET_ACCESS_KEY=$(AWS_SECRET_ACCESS_KEY)             -e GITHUB_TOKEN=$(GITHUB_TOKEN)             -e ghprbPullId=$(ghprbPullId)             -v .jenkins/test-result:/result             modin-project/py3-test:$(sha_tag)) 2>&1 | python3 ./jenkins/utils/colorize_output.py --tag test_py3-test_Python_4
	
		@echo ===== finished: test_py3-test_Python_4 =====
	
        

all: make_result_dir publish_report build_py2-test build_py3-test test_py2-test_Ray_4 test_py2-test_Ray_8 test_py2-test_Ray_12 test_py2-test_Ray_16 test_py2-test_Dask_4 test_py2-test_Python_4 test_py3-test_Ray_4 test_py3-test_Ray_8 test_py3-test_Ray_12 test_py3-test_Ray_16 test_py3-test_Dask_4 test_py3-test_Python_4


build: build_py2-test build_py3-test


test: test_py2-test_Ray_4 test_py2-test_Ray_8 test_py2-test_Ray_12 test_py2-test_Ray_16 test_py2-test_Dask_4 test_py2-test_Python_4 test_py3-test_Ray_4 test_py3-test_Ray_8 test_py3-test_Ray_12 test_py3-test_Ray_16 test_py3-test_Dask_4 test_py3-test_Python_4

Hiii!
SHELL=/bin/bash -o pipefail
sha_tag=$(shell git rev-parse --verify --short HEAD)
sha_tag=$(shell git rev-parse --verify --short HEAD)

placeholder: 
	echo "Do not run make without any target!"
        
Hiii!
