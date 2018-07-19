#!/bin/sh

HOSTNAME=$1
KEY=$2
PORT=$3

ssh -i $2 -L $PORT:localhost:$PORT $1 "bash -s" << INT
PATH=$PATH:~/.local/bin/    # ensure Jupyter is in the path
jupyter notebook --port=$PORT
