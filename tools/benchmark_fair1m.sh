#!/usr/bin/env bash

CONFIG_NAME=$1
PORT=${PORT:-29507}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

WORK_DIR='work_dirs/'$CONFIG_NAME
CONFIG=$WORK_DIR'/'$CONFIG_NAME'.py'
CHECKPOINT=$WORK_DIR'/latest.pth'

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=1 \
    --master_port=$PORT \
    tools/analysis_tools/benchmark.py \
    $CONFIG \
    $CHECKPOINT \
    --launcher pytorch \
    ${@:4}
