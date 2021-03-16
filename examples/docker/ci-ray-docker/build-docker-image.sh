#!/bin/bash -e

# Licensed to Modin Development Team under one or more contributor license agreements.
# See the NOTICE file distributed with this work for additional information regarding
# copyright ownership.  The Modin Development Team licenses this file to you under the
# Apache License, Version 2.0 (the "License"); you may not use this file except in
# compliance with the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under
# the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.

script_dir="`dirname \"$0\"`"
pushd $script_dir
modin_root=$(readlink -f ../../../)
cp $modin_root/modin ./modin -r
cp $modin_root/requirements/requirements-no-engine.yml ./requirements-no-engine.yml
docker build -f ci-ray-docker.dockerfile -t ray_docker_image:1.1.0 .
rm -rf ./modin
rm -f ./requirements-no-engine.yml
popd
