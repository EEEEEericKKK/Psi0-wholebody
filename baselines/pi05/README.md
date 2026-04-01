## OpenPI $\pi_{0.5}$ 


### Set-up Envrionment
> ℹ️ The following commands assume the variables eg., `PSI_HOME` are loaded by running `cd /path/to/psi0 &&  source .env`.


Set up seperate environment for $\pi_{0.5}$: 

> ℹ️ We manage the $\Psi_0$ environment and all the baselines through `uv` and they all share the same `src/` code.  See [Environment Management](../README.md) for more details.

```
uv venv .venv-openpi --python 3.10
source .venv-openpi/bin/activate
VIRTUAL_ENV=.venv-openpi uv pip install -e .
VIRTUAL_ENV=.venv-openpi uv pip install -e src/openpi/openpi-client
VIRTUAL_ENV=.venv-openpi GIT_LFS_SKIP_SMUDGE=1 uv pip install -r baselines/pi05/requirements-openpi.txt
```

Apply the `transformers` library patches:

> See also the official OpenPI [README.md](https://github.com/Physical-Intelligence/openpi?tab=readme-ov-file#setup)

```
cp -r src/openpi/models_pytorch/transformers_replace/* .venv-openpi/lib/python3.10/site-packages/transformers/
```

### Download Pre-Trained Weights

Download pretrained `pi05_droid` model of `torch` variant:

```
hf download USC-PSI-Lab/psi-model \
	--local-dir=$PSI_HOME/cache/checkpoints \
	--include="openpi/pi05_droid/*" \
	--repo-type=model
```

<details>
<summary>[Optional] Expand to see how we produced the `pi05_droid` pytorch checkpoint.</summary>

	Download `pi05-droid` checkpoints

	```
	python src/openpi/shared/download.py
	```

	Convet from `jax` to `pytorch` checkpoint
	```
	uv run examples/convert_jax_model_to_pytorch.py \
		--checkpoint_dir=/home/user/.cache/openpi/openpi-assets/checkpoints/pi05_droid \
		--config_name=pi05_droid \
		--output_path $PSI_HOME/cache/checkpoints/openpi/pi05_droid
	```
</details>

### Download Psi0 Task Data

Download the task data, for example
```
export task=G1WholebodyBendPick-v1
```
```
hf download songlinwei/hfm simple/$task.zip --local-dir=$PSI_HOME/data --repo-type=dataset
unzip "$PSI_HOME/data/simple/$task.zip" -d "$PSI_HOME/data/simple"
```
Create a new `TrainConfig` for the task in `src/openpi/training/config.py`:
```
vim src/openpi/training/config.py
```
for example
```
	# ...
	TrainConfig(
        name="simple_bend_pick_v1",
        project_name="psi",
        num_workers=8,
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=36,
            action_horizon=30,
            max_token_len=250,
        ),
        data=LeRobotHFMDataConfig(
            repo_id=f"{os.environ['PSI_HOME']}/data/simple/G1WholebodyBendPick-v1",
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_droid/params"),
        num_train_steps=40_000,
        batch_size=128,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=1e-4,
            decay_steps=40_000,
            decay_lr=1e-8,
        ),
        pytorch_weight_path=f"{os.environ['PSI_HOME']}/cache/checkpoints/pi05_droid",
        policy_metadata={"dataset": "G1WholebodyBendPick-v1"},
        checkpoint_base_dir=f".runs/openpi-05"
    ),
	# ...
```

Compute stats for the task using official script (which is slow)
```
python src/openpi/compute_norm_stats.py --config-name simple_bend_pick_v1
```

> Or you can rewrite our precomputed stats to `openpi` format:
> `python src/openpi/rewrite_norm_stats.py --task_path=$PSI_HOME/data/simple/G1WholebodyBendPick-v1`.
>  The calculations are slightly different but it's ok.

### Fine-Tune $\pi_{0.5}$
Launch the training script
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nnodes=1 --nproc_per_node=8 src/openpi/train_pytorch.py \
	simple_bend_pick_v1 \
	--exp_name=simple_bend_pick_v1 \
	--save_interval=10000 \
	--checkpoint_base_dir=.runs/openpi-05
```


### Eval $\pi_{0.5}$
```
export port=9000
bash scripts/deploy/serve_pi05.sh $task $step $port
```
Open-loop evaluation
```
export PYTORCH_WEIGHT_PATH=/hfm/songlin/we_learn/.cache/checkpoints/pi05_droid
export DATA_HOME=/hfm/songlin/we_learn/.data/real_teleop_g1/lerobot
export task=Pick_bottle_and_turn_and_pour_into_cup
export port=9000

python scripts/train/openpi/eval_openloop.py --port=$port --task=$task
```


<details>
<summary>[Optional] Expand to see more implementation details.</summary>

Below are the gist of how to adpat OpenPI on humaonoid loco-manipulation tasks:

1. change action dim from `32` to `36`
	```
	# vim src/openpi/models_pytorch/pi0_pytorch.py
	self.action_in_proj = nn.Linear(36, action_expert_config.width)
	self.action_out_proj = nn.Linear(action_expert_config.width, 36)
	```

2. adapt the loading of pretrained `pi05_torch`
	```
	model_path = os.path.join(config.pytorch_weight_path, "model.safetensors")
	# Psi-0: adapt action dim to 36
	from safetensors.torch import load_file
	state_dict = load_file(model_path)
	pad_dim = config.model.action_dim - state_dict["action_in_proj.weight"].shape[1]
	if pad_dim > 0:
		# eg., torch.Size([1024, 32]) -> torch.Size([1024, 36])
		# Replicate the last 4 columns instead of padding with zeros
		w = state_dict["action_in_proj.weight"]
		to_pad = w[:, -pad_dim:]
		state_dict["action_in_proj.weight"] = torch.cat([w, to_pad], dim=1)

		b = state_dict["action_out_proj.bias"]
		state_dict["action_out_proj.bias"] = torch.cat([b, b[-pad_dim:]], dim=0)

		w = state_dict["action_out_proj.weight"]
		to_pad = w[-pad_dim:, :]
		state_dict["action_out_proj.weight"] = torch.cat([w, to_pad], dim=0)

	# https://github.com/Physical-Intelligence/openpi/issues/669
	state_dict["paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"] = \
		state_dict["paligemma_with_expert.paligemma.lm_head.weight"]

	_model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
	missing_keys, unexpected_keys = _model.load_state_dict(state_dict, strict=False)
	```

3. create TrainConfig in config.py
	```
	TrainConfig(
        name="Pick_toys_into_box_and_lift_and_turn_and_put_on_the_chair_new_target_yaw",
        project_name="hfm",
        num_workers=8,
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=36,
            action_horizon=16,
            max_token_len=250,
        ),
        data=LeRobotHFMDataConfig( # FIXME
            repo_id= f"{os.environ['DATA_HOME']}/Pick_toys_into_box_and_lift_and_turn_and_put_on_the_chair_new_target_yaw",
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_droid/params"),
        num_train_steps=40_000,
        batch_size=128,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=1e-4,
            decay_steps=40_000,
            decay_lr=1e-8,
        ),
        pytorch_weight_path=os.environ["PYTORCH_WEIGHT_PATH"],
        policy_metadata={"dataset": "Pick_toys_into_box_and_lift_and_turn_and_put_on_the_chair_new_target_yaw"},
    ),
	```

4. launch training
```
bash scripts/train/openpi/benchmark_pi05_nv_slurm.sh \
	Pick_toys_into_box_and_lift_and_turn_and_put_on_the_chair_new_target_yaw
```

5. [Optional] upload model weights

```
#export task=Hold_lunch_bag_with_both_hands_and_squat_to_put_on_the_coffee_table
#export task=Pick_toys_into_box_and_lift_and_turn_and_put_on_the_chair_new_target_yaw
export task=Pull_the_tray_out_of_chips_can_and_throw_the_can_into_trash_bin
export step=40000
hf upload USC-PSI-Lab/psi-model \
	.runs/openpi-05/$task/$task/$step/model.safetensors \
	benchmarks/openpi-05/$task/$step/model.safetensors \
	--repo-type=model
```


6. Serve
Download:
```
export task=Remove_the_cap_turn_on_the_faucet_and_fill_the_bottle_with_water
export step=40000
python scripts/data/download.py \
	--repo-id=USC-PSI-Lab/psi-data-models \
	--remote-dir=benchmarks/openpi-05/$task/$step \
	--repo-type=model \
	--local-dir=.runs/openpi-05/$task/$task/$step
```
and Serve:
```
export port=9000
bash scripts/deploy/serve_pi05.sh $task $step $port
```
Open-loop evaluation
```
export PYTORCH_WEIGHT_PATH=/hfm/songlin/we_learn/.cache/checkpoints/pi05_droid
export DATA_HOME=/hfm/songlin/we_learn/.data/real_teleop_g1/lerobot
export task=Pick_bottle_and_turn_and_pour_into_cup
export port=9000

python scripts/train/openpi/eval_openloop.py --port=$port --task=$task
```
</details>