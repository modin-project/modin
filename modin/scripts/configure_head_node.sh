#!/bin/sh

HOSTNAME=$1
KEY=$2

ssh -i $2 -o "StrictHostKeyChecking no" $1 << "ENDSSH"
python -m pip install modin jupyter
PATH=$PATH:~/.local/bin/    # ensure Ray is in the path
ray stop
ray start --head
ENDSSH
