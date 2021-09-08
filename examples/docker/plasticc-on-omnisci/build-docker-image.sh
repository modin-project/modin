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

cd "`dirname \"$0\"`"

docker build -f plasticc-omnisci.dockerfile -t plasticc-omnisci --build-arg no_proxy \
    --build-arg https_proxy --build-arg http_proxy .

echo -e "\nNote: a user is responsible for preparing the datasets.
The datasets must include four files: training set, test set,
training set metadata and test set metadata."
printf "\n\nTo run the benchmark execute:\n"
printf "\tdocker run --rm -v /path/to/dataset:/dataset plasticc-omnisci <training set file name in /path/to/dataset> <test set file name in /path/to/dataset> <training set metadata file name in /path/to/dataset> <test set metadata file name in /path/to/dataset>\n"
