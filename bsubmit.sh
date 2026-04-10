#!/bin/bash

# usage: bv-submit.sh <path_to_script> [num_nodes] [num_gpus]
script=$1
num_nodes=${2:-1}
num_gpus=${3:-8}
echo "script: $script, num_nodes: $num_nodes, num_gpus: $num_gpus"

export job_name=granite_h2vlm

bsub \
        -J $job_name \
        -gpu "num=$num_gpus/task:j_exclusive=yes:mode=shared" \
        -n $num_nodes \
        -M 2048G \
        -W 60:00 \
        -G grp_inference_scaling \
        -o /proj/inf-scaling/h2vla/bsub_outputs/${job_name}-%J.stdout \
        -e /proj/inf-scaling/h2vla/bsub_outputs/${job_name}-%J.stderr \
        -R "select[!pod6]" \
        blaunch sh $script \

exit 0