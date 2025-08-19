#!/bin/bash
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export CUDA_LAUNCH_BLOCKING=1

FORCE_TORCHRUN=1
NPROC_PER_NODE=4  
NNODES=1          
RANK=0
MASTER_ADDR=127.0.0.1
MASTER_PORT=2349

export CUDA_VISIBLE_DEVICES=0,1,2,3

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "NPROC_PER_NODE=$NPROC_PER_NODE"
echo "NNODES=$NNODES"
echo "RANK=$RANK"
echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"

torchrun \
    --nproc_per_node $NPROC_PER_NODE \
    --nnodes $NNODES \
    --node_rank $RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    #your train.py    #your yaml file
