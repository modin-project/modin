#!/usr/bin/sh

HOSTNAME=$1
NODE_LIST_FILEPATH=$2
NOTEBOOK_PORT=$3

HEAD_NODE_CONFIG=".modin_head_node_setup.sh"
NODE_LIST_FILE=$(basename $NODE_LIST_FILEPATH)
REDIS_PORT=6379

cat > $HEAD_NODE_CONFIG <<- EOM
PATH=\$PATH:~/.local/bin/    # ensure Ray is in the path

echo "Attempting to connect to worker nodes"
for HOST in \$(cat $NODE_LIST_FILE); do
    echo -n "Connecting to $HOST..."
    ssh -o "StrictHostKeyChecking no" $HOST uptime
    echo "Connected"
done

if command -v ray; then
    ray stop
fi

echo "Installing python dependencies..."
python -m pip install modin jupyter

echo "Starting Ray on port $REDIS_PORT..."
ray start --head --redis-port=$REDIS_PORT

echo "Connecting worker nodes..."
HEAD_NODE_IP=\$(hostname -I | tr -d "[:space:]")
REDIS_ADDRESS="\$HEAD_NODE_IP:$REDIS_PORT"

for HOST in \$(cat $NODE_LIST_FILE); do
    echo "Starting Ray on \$HOST"
    ssh -o StrictHostKeyChecking=no \$HOST "python -m pip install modin; PATH=\\\$PATH:~/.local/bin; ray start --redis-address=\$REDIS_ADDRESS" > /dev/null &
done

if [ "${NOTEBOOK_PORT-}" ]; then
    MODIN_EXECUTION_FRAMEWORK=ray MODIN_RAY_REDIS_ADDRESS=\$REDIS_ADDRESS jupyter notebook --port $NOTEBOOK_PORT
fi
EOM

scp $NODE_LIST_FILEPATH $HOSTNAME:~
scp $HEAD_NODE_CONFIG $HOSTNAME:~

if [ "${NOTEBOOK_PORT-}" ]; then
    ssh -A -L $NOTEBOOK_PORT:localhost:$NOTEBOOK_PORT -o StrictHostKeyChecking=no $HOSTNAME "bash $HEAD_NODE_CONFIG"
else
    ssh -A -o StrictHostKeyChecking=no $HOSTNAME "bash $HEAD_NODE_CONFIG"
fi
