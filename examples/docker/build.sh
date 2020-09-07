#!/bin/bash -e

cd "`dirname \"$0\"`"

docker build -f nyc-taxi.dockerfile -t nyc-taxi --build-arg https_proxy --build-arg http_proxy .
printf "\n\nTo run the benchmark execute:\n\tdocker run --rm nyc-taxi\n"
