#!/bin/sh

HOSTNAME=$1
KEY=$2
REDIS_ADDRESS=$3

ssh -i $2 -o "StrictHostKeyChecking no" $1 REDIS_ADDRESS=$REDIS_ADDRESS "bash -s" << "ENDSSH"
python -m pip install modin
PATH=$PATH:~/.local/bin/    # ensure Ray is in the path
ray stop
ray start --redis-address $REDIS_ADDRESS
ENDSSH
