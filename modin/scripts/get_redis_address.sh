#!/bin/sh

STRING=$1

RAY_START_CMD=$(echo $STRING | grep -o "ray start --redis-address [0-9\.:]\+")
REDIS_ADDRESS=$(echo $RAY_START_CMD | grep -o "[0-9\.:]\+")
echo $REDIS_ADDRESS
