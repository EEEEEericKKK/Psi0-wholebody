#!/usr/bin/env bash

# Common setup script for LlamaFactory training jobs
# Source this script in your training scripts to avoid code duplication

CLEAN_HOST_LIST=$(echo "$LSB_MCPU_HOSTS" | awk '{for(i=1; i<=NF; i+=2) printf "%s ", $i}')
export NNODES=$(echo "$CLEAN_HOST_LIST" | wc -w)
export MASTER_ADDR=$(echo "$CLEAN_HOST_LIST" | awk '{print $1}')
export NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}
CURRENT_HOST=$(hostname -s)
HOST_ARRAY=($CLEAN_HOST_LIST)
export NODE_RANK=-1
for i in "${!HOST_ARRAY[@]}"; do
   if [[ "${HOST_ARRAY[$i]}" == "$CURRENT_HOST" ]]; then
       export NODE_RANK=$i
       break
   fi
done

# Error out if the rank could not be determined (in multinode setup).
if [ "$NNODES" -gt 1 ] && [ "$NODE_RANK" -eq -1 ]; then
    echo "ERROR: Could not determine NODE_RANK." >&2
    echo "  - Current (short) Host: $CURRENT_HOST" >&2
    echo "  - Clean Host List Searched: ${CLEAN_HOST_LIST}" >&2
    echo "  - Original LSF Host List: $LSB_MCPU_HOSTS" >&2
    exit 1
fi

# Define a static port for the master node to listen on.
export MASTER_PORT=${MASTER_PORT:-29500}

eval "$(conda shell.bash hook)"

export PYTHONNOUSERSITE=1
export CUDA_HOME=/proj/inf-scaling/vlmcanvas/.env
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export HF_HOME=/proj/inf-scaling/huggingface
export TMPDIR=/proj/inf-scaling/TMP

export WANDB_API_KEY=55e59d4db1f11a22713ac08a884b1b44ce20caf2
export WANDB_PROJECT=psi0-wholebody