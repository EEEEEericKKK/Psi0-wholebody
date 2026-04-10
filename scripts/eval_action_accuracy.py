#!/usr/bin/env python3
"""
Calculate action prediction accuracy on training/validation set.

This script evaluates a trained model's action predictions against ground truth
and computes various accuracy metrics.

Usage:
    python scripts/eval_action_accuracy.py \
        --checkpoint path/to/checkpoint.pth \
        --config posttrain_he_psi0_config \
        --split val \
        --num_samples 1000
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class ActionAccuracyEvaluator:
    """Evaluate action prediction accuracy."""
    
    def __init__(
        self,
        model,
        dataset,
        device: str = "cuda",
        action_dim: int = 14,
    ):
        self.model = model
        self.dataset = dataset
        self.device = device
        self.action_dim = action_dim
        
    def compute_metrics(
        self,
        pred_actions: np.ndarray,
        gt_actions: np.ndarray,
        action_mask: np.ndarray = None,
    ) -> Dict[str, float]:
        """
        Compute various accuracy metrics.
        
        Args:
            pred_actions: Predicted actions (B, T, D)
            gt_actions: Ground truth actions (B, T, D)
            action_mask: Mask for valid actions (B, T, D)
            
        Returns:
            Dictionary of metrics
        """
        if action_mask is not None:
            # Apply mask
            pred_actions = pred_actions * action_mask
            gt_actions = gt_actions * action_mask
            valid_count = action_mask.sum()
        else:
            valid_count = pred_actions.size
            
        # Mean Squared Error
        mse = np.mean((pred_actions - gt_actions) ** 2)
        
        # Mean Absolute Error
        mae = np.mean(np.abs(pred_actions - gt_actions))
        
        # Root Mean Squared Error
        rmse = np.sqrt(mse)
        
        # Per-dimension metrics
        per_dim_mae = np.mean(np.abs(pred_actions - gt_actions), axis=(0, 1))
        per_dim_mse = np.mean((pred_actions - gt_actions) ** 2, axis=(0, 1))
        
        # Accuracy within thresholds (for position/rotation)
        # Threshold for position: 5cm, rotation: 0.1 rad
        pos_threshold = 0.05  # 5cm
        rot_threshold = 0.1   # ~5.7 degrees
        
        # Assume first 12 dims are positions/rotations
        if self.action_dim >= 12:
            # Left hand XYZ (0-2), RPY (3-5)
            # Right hand XYZ (6-8), RPY (9-11)
            pos_errors = np.abs(pred_actions[:, :, [0, 1, 2, 6, 7, 8]] - 
                               gt_actions[:, :, [0, 1, 2, 6, 7, 8]])
            rot_errors = np.abs(pred_actions[:, :, [3, 4, 5, 9, 10, 11]] - 
                               gt_actions[:, :, [3, 4, 5, 9, 10, 11]])
            
            pos_accuracy = (pos_errors < pos_threshold).mean() * 100
            rot_accuracy = (rot_errors < rot_threshold).mean() * 100
        else:
            pos_accuracy = 0.0
            rot_accuracy = 0.0
        
        # Overall accuracy (within combined threshold)
        overall_threshold = 0.1
        overall_accuracy = (np.abs(pred_actions - gt_actions) < overall_threshold).mean() * 100
        
        metrics = {
            "mse": float(mse),
            "mae": float(mae),
            "rmse": float(rmse),
            "pos_accuracy_5cm": float(pos_accuracy),
            "rot_accuracy_5deg": float(rot_accuracy),
            "overall_accuracy_thresh": float(overall_accuracy),
            "per_dim_mae": per_dim_mae.tolist(),
            "per_dim_mse": per_dim_mse.tolist(),
            "valid_samples": int(valid_count / self.action_dim),
        }
        
        return metrics
    
    @torch.no_grad()
    def evaluate(
        self,
        num_samples: int = None,
        batch_size: int = 1,
    ) -> Dict[str, Any]:
        """
        Evaluate model on dataset.
        
        Args:
            num_samples: Number of samples to evaluate (None = all)
            batch_size: Batch size for evaluation
            
        Returns:
            Dictionary of evaluation results
        """
        self.model.eval()
        
        num_samples = num_samples or len(self.dataset)
        num_samples = min(num_samples, len(self.dataset))
        
        all_pred_actions = []
        all_gt_actions = []
        all_masks = []
        
        print(f"Evaluating {num_samples} samples...")
        
        for idx in tqdm(range(num_samples)):
            try:
                # Get sample
                sample = self.dataset[idx]
                
                # Move to device and add batch dimension
                batch = {}
                for key, value in sample.items():
                    if isinstance(value, torch.Tensor):
                        batch[key] = value.unsqueeze(0).to(self.device)
                    elif isinstance(value, np.ndarray):
                        batch[key] = torch.from_numpy(value).unsqueeze(0).to(self.device)
                    else:
                        batch[key] = value
                
                # Forward pass
                pred_actions = self.model(batch)
                
                # Extract ground truth actions
                if "actions" in batch:
                    gt_actions = batch["actions"]
                elif "raw_actions" in batch:
                    gt_actions = batch["raw_actions"]
                else:
                    print(f"Warning: No ground truth actions found in sample {idx}")
                    continue
                
                # Extract mask if available
                action_mask = batch.get("actions_mask", None)
                
                # Convert to numpy
                pred_actions_np = pred_actions.cpu().numpy()
                gt_actions_np = gt_actions.cpu().numpy()
                mask_np = action_mask.cpu().numpy() if action_mask is not None else None
                
                all_pred_actions.append(pred_actions_np)
                all_gt_actions.append(gt_actions_np)
                if mask_np is not None:
                    all_masks.append(mask_np)
                    
            except Exception as e:
                print(f"Error processing sample {idx}: {e}")
                continue
        
        # Concatenate all predictions
        all_pred_actions = np.concatenate(all_pred_actions, axis=0)
        all_gt_actions = np.concatenate(all_gt_actions, axis=0)
        all_masks = np.concatenate(all_masks, axis=0) if all_masks else None
        
        print(f"Predictions shape: {all_pred_actions.shape}")
        print(f"Ground truth shape: {all_gt_actions.shape}")
        
        # Compute metrics
        metrics = self.compute_metrics(all_pred_actions, all_gt_actions, all_masks)
        
        # Add dimension names for 16D hands_only format
        if self.action_dim == 16:
            dim_names = [
                "left_x", "left_y", "left_z",
                "left_roll", "left_pitch", "left_yaw",
                "right_x", "right_y", "right_z",
                "right_roll", "right_pitch", "right_yaw",
                "head_x", "head_y", "head_z", "discrete_token"
            ]
            metrics["dimension_names"] = dim_names
        
        return metrics


def load_config_from_checkpoint(checkpoint_path: str):
    """Load configuration from checkpoint directory."""
    checkpoint_dir = Path(checkpoint_path).parent
    
    # Try to find config.json in checkpoint directory
    config_paths = [
        checkpoint_dir / "config.json",
        checkpoint_dir.parent / "config.json",
    ]
    
    for config_path in config_paths:
        if config_path.exists():
            print(f"Loading config from: {config_path}")
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            return config_dict
    
    print("⚠ Warning: No config.json found, using command line arguments")
    return None


def main():
    parser = argparse.ArgumentParser(description="Evaluate action prediction accuracy")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (.pth file) - optional for data-only evaluation"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="posttrain_he_psi0_config",
        help="Configuration name"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Data root directory (default: from env DATA_HOME)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val"],
        help="Dataset split to evaluate"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Number of samples to evaluate (default: 1000)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--action-dim",
        type=int,
        default=16,
        help="Action dimension (16 or 36)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="action_accuracy_results.json",
        help="Output JSON file for metrics"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for evaluation"
    )
    parser.add_argument(
        "--action-format",
        type=str,
        default="hands_only",
        choices=["full", "hands_only"],
        help="Action format (hands_only=16D, full=36D)"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Action Accuracy Evaluation")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint or 'None (data-only mode)'}")
    print(f"Config: {args.config}")
    print(f"Split: {args.split}")
    print(f"Num samples: {args.num_samples}")
    print(f"Action format: {args.action_format} ({args.action_dim}D)")
    print(f"Device: {args.device}")
    print("=" * 80)
    
    # Check checkpoint exists (if provided)
    if args.checkpoint and not Path(args.checkpoint).exists():
        print(f"❌ Error: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    
    # Load configuration from checkpoint if available
    print("\n[1/5] Loading configuration...")
    if args.checkpoint:
        config_dict = load_config_from_checkpoint(args.checkpoint)
    else:
        print("Running in data-only mode (no checkpoint)")
        config_dict = None
    
    # Get data root
    data_root = args.data_root or os.environ.get("DATA_HOME")
    if data_root:
        data_root = Path(data_root) / "HE_RAW"
    else:
        print("❌ Error: DATA_HOME not set and --data-root not provided")
        sys.exit(1)
    
    if not data_root.exists():
        print(f"❌ Error: Data directory not found: {data_root}")
        sys.exit(1)
    
    print(f"✓ Data root: {data_root}")
    
    # Import and create dataset
    print("\n[2/5] Loading dataset...")
    try:
        from src.psi.data.humanoid.he_raw_dataset import HERawDataset
        from src.psi.config.transform import HEPosttrainRepackTransform, ActionStateTransform, Psi0ModelTransform
        from transformers import AutoProcessor
        
        # Create transforms
        repack_transform = HEPosttrainRepackTransform(
            action_chunk_size=16,
            use_delta_actions=True,
            pad_action_dim=args.action_dim,
            pad_state_dim=args.action_dim,
            action_format=args.action_format,
        )
        
        # Load stats for normalization
        if args.action_format == "hands_only":
            stats_path = "assets/stats/he_raw_rel_stats_16d.json"
        else:
            stats_path = "assets/stats/he_raw_rel_stats_combined_no_static.json"
        
        field_transform = ActionStateTransform(
            action_norm_type="bounds_q99",
            stat_path=stats_path,
            normalize_state=False,
            pad_action_dim=args.action_dim,
            pad_state_dim=args.action_dim,
        )
        
        vlm_processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen3-VL-2B-Instruct",
            min_pixels=16 * 28 * 28,
            max_pixels=576 * 28 * 28,
        )
        
        model_transform = Psi0ModelTransform(
            img_aug=False,
        )
        
        # Create dataset
        dataset = HERawDataset(
            data_root=str(data_root),
            num_past_frames=0,
            action_chunk_size=17,  # 16 + 1 for delta
            use_delta_actions=True,
            robot_type="both",
        )
        
        print(f"✓ Dataset loaded: {len(dataset)} samples")
        print(f"✓ Evaluating first {min(args.num_samples, len(dataset))} samples")
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Load model
    print("\n[3/5] Loading model...")
    print("⚠ Warning: Model loading from checkpoint is not fully implemented")
    print("⚠ This script computes metrics on data preprocessing only")
    print("⚠ To evaluate a trained model, you need to add model loading logic")
    
    # For now, compute metrics on transformed data (data accuracy)
    print("\n[4/5] Computing data preprocessing metrics...")
    
    all_actions = []
    all_states = []
    sample_count = 0
    
    for idx in tqdm(range(min(args.num_samples, len(dataset)))):
        try:
            # Get raw sample
            sample = dataset[idx]
            
            # Apply transforms
            transformed = repack_transform(sample)
            transformed = field_transform(transformed)
            transformed = model_transform(transformed, vlm_processor=vlm_processor)
            
            # Collect actions
            actions = transformed.get("actions")
            states = transformed.get("states")
            
            if actions is not None:
                if isinstance(actions, torch.Tensor):
                    actions = actions.numpy()
                all_actions.append(actions)
                sample_count += 1
            
            if states is not None:
                if isinstance(states, torch.Tensor):
                    states = states.numpy()
                all_states.append(states)
                
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            continue
    
    # Compute statistics
    print("\n[5/5] Computing statistics...")
    
    all_actions = np.concatenate(all_actions, axis=0)
    print(f"✓ Processed {sample_count} samples")
    print(f"✓ Actions shape: {all_actions.shape}")
    
    # Compute data statistics
    results = {
        "config": {
            "checkpoint": str(args.checkpoint) if args.checkpoint else "None",
            "split": args.split,
            "num_samples": sample_count,
            "action_dim": args.action_dim,
            "action_format": args.action_format,
        },
        "data_statistics": {
            "action_mean": all_actions.mean(axis=0).tolist(),
            "action_std": all_actions.std(axis=0).tolist(),
            "action_min": all_actions.min(axis=0).tolist(),
            "action_max": all_actions.max(axis=0).tolist(),
            "per_dim_stats": [],
        }
    }
    
    # Per-dimension statistics
    if args.action_dim == 16:
        dim_names = [
            "left_x", "left_y", "left_z",
            "left_roll", "left_pitch", "left_yaw",
            "right_x", "right_y", "right_z",
            "right_roll", "right_pitch", "right_yaw",
            "head_x", "head_y", "head_z", "discrete_token"
        ]
        
        for i, name in enumerate(dim_names):
            results["data_statistics"]["per_dim_stats"].append({
                "dimension": i,
                "name": name,
                "mean": float(all_actions[:, i].mean()),
                "std": float(all_actions[:, i].std()),
                "min": float(all_actions[:, i].min()),
                "max": float(all_actions[:, i].max()),
            })
    
    # Save results
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 80)
    print("Evaluation complete!")
    print("=" * 80)
    print(f"\n📊 Results Summary:")
    print(f"   Samples processed: {sample_count}")
    print(f"   Action shape: {all_actions.shape}")
    print(f"   Action range: [{all_actions.min():.3f}, {all_actions.max():.3f}]")
    print(f"   Action mean: {all_actions.mean():.3f} ± {all_actions.std():.3f}")
    print(f"\n💾 Results saved to: {output_path}")
    print("\n⚠ Note: This script currently computes data preprocessing metrics.")
    print("   To evaluate model predictions, add model loading and inference logic.")


if __name__ == "__main__":
    main()
