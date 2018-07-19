#!/bin/sh

HOSTNAME=$1
KEY=$2
PORT=$3
EXECUTION_FRAMEWORK=$4
RAY_REDIS_ADDRESS=$5

ssh -i $2 -L $PORT:localhost:$PORT $1 "bash -s" << INT
PATH=$PATH:~/.local/bin/    # ensure Jupyter is in the path
MODIN_EXECUTION_FRAMEWORK=$EXECUTION_FRAMEWORK \
    MODIN_RAY_REDIS_ADDRESS=$RAY_REDIS_ADDRESS \
    jupyter notebook --port=$PORT
