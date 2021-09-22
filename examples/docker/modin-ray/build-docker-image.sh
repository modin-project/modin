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

docker build -t modin-ray .

echo -e '\nNYC TAXI BENCHMARK
User is responsible for preparing the dataset.
It Can be generated by following the instructions on the link:
https://github.com/toddwschneider/nyc-taxi-data#instructions
To run the benchmark execute:
\tdocker run --rm -v /path/to/dataset:/dataset python nyc-taxi.py <name of file starting with /dataset>

CENSUS BENCHMARK
User is responsible for preparing the dataset.
It can be downloaded from the following link:
https://rapidsai-data.s3.us-east-2.amazonaws.com/datasets/ipums_education2income_1970-2010.csv.gz
To run the benchmark execute:
\tdocker run --rm -v /path/to/dataset:/dataset python census.py <name of file starting with /dataset>

PLASTICC BENCHMARK
User is responsible for preparing the datasets.
The datasets must include four files: training set, test set,
training set metadata and test set metadata.
To run the benchmark execute:
\tdocker run --rm -v /path/to/dataset:/dataset python plasticc.py <training set file name starting with /dataset> <test set file name starting with /dataset> <training set metadata file name starting with /dataset> <test set metadata file name starting with /dataset>\n'
