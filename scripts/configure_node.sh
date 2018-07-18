#!/bin/sh

HOSTNAME=$1
KEY=$2
RAY_START_CMD=$3

ssh -i $2 -o "StrictHostKeyChecking no" $1 RAY_START_CMD=$RAY_START_CMD "bash -s" << "ENDSSH"
pip3 install modin jupyter
ray stop
$RAY_START_CMD
ENDSSH
