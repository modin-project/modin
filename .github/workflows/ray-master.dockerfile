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

FROM rayproject/ray:nightly
WORKDIR /home/ray
COPY ./modin ./modin
COPY requirements/requirements-no-engine.yml ./requirements-no-engine.yml
RUN sudo chown ray:users ./modin -R && sudo chown ray:users ./requirements-no-engine.yml
RUN sudo apt-get update --yes \
    && sudo apt-get install -y libhdf5-dev
RUN conda env update -f requirements-no-engine.yml --name base

CMD ["/bin/bash"]
