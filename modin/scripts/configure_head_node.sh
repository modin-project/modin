#!/bin/sh

HOSTNAME=$1
KEY=$2

ssh -i $2 -o "StrictHostKeyChecking no" $1 << "ENDSSH"
pip3 install modin jupyter
ray stop
ray start --head
ENDSSH
