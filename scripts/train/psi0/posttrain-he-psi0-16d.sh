#!/bin/bash

set -e

cd /proj/inf-scaling/h2vla/wholebodyVLA/Psi0-wholebody

source /proj/inf-scaling/h2vla/wholebodyVLA/Psi0-wholebody/common_setup.sh
source /proj/inf-scaling/h2vla/wholebodyVLA/Psi0-wholebody/.env
source /proj/inf-scaling/h2vla/wholebodyVLA/Psi0-wholebody/.venv-psi/bin/activate

export OMP_NUM_THREADS=8
export TF_CPP_MIN_LOG_LEVEL=3

args="posttrain_he_psi0_config \
--seed=292285 \
--exp=posttrain_16d \
--timestamp=$(date +"%y%m%d%H%M") \
--train.name=posttrain \
--train.data_parallel=ddp \
--train.mixed_precision=bf16 \
--train.train_batch_size=64 \
--train.resume_from_checkpoint=latest \
--train.max_checkpoints_to_keep=5 \
--train.gradient_accumulation_steps=1 \
--train.learning_rate=1e-4 \
--train.max_training_steps=10000 \
--train.warmup_ratio=None \
--train.warmup_steps=100 \
--train.checkpointing_steps=2000 \
--train.validation_steps=1000 \
--train.val_num_batches=20 \
--train.max_grad_norm=1.0 \
--train.lr_scheduler_type=constant \
--train.lr_scheduler_kwargs.betas 0.9 0.999 \
--train.lr_scheduler_kwargs.weight_decay=0.0 \
--train.lr_scheduler_kwargs.eps=1e-8 \
--log.report_to=wandb \
--data.root-dir=$DATA_HOME/HE_RAW \
--data.use-delta-actions \
--data.transform.repack.action-chunk-size=16 \
--data.transform.repack.use-delta-actions \
--data.transform.repack.action-format=hands_only \
--data.transform.repack.pad-action-dim=16 \
--data.transform.repack.pad-state-dim=16 \
--data.transform.field.action_norm_type=bounds_q99 \
--data.transform.field.stat-path=assets/stats/he_raw_rel_stats_16d.json \
--data.transform.field.no-normalize-state \
--data.transform.field.pad-action-dim=16 \
--data.transform.field.pad-state-dim=16 \
--data.transform.model.resize.size 240 320 \
--data.transform.model.center_crop.size 240 320 \
--data.transform.model.no-img-aug \
--model.model_name_or_path=/proj/inf-scaling/h2vla/data/huggingface/checkpoints/psi0/pre.fast.egodex.2512241941.ckpt200k \
--model.noise-scheduler=flow \
--model.n_conditions=0 \
--model.action-chunk-size=16 \
--model.action-dim=16 \
--model.action-exec-horizon=16 \
--model.observation-horizon=1 \
--model.odim=16 \
--model.view_feature_dim=2048 \
--model.no-tune-vlm \
--model.no-use_film \
--model.no-combined_temb
"

torchrun --nproc_per_node=$NGPUS_PER_NODE --master_port=$MASTER_PORT --nnodes=$NNODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR scripts/train.py \
    ${args}
