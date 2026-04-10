# Action Format 14D Documentation

This directory contains comprehensive documentation for the 14D action format modification to the Psi0 VLA action expert.

## Quick Links

### Start Here
- **[MODIFICATION_SUMMARY.md](./MODIFICATION_SUMMARY.md)** - Quick overview and usage guide
- **[TRAINING_VERIFICATION.md](./TRAINING_VERIFICATION.md)** - Proof that it works!

### Detailed Documentation
- **[ACTION_FORMAT_14D_README.md](./ACTION_FORMAT_14D_README.md)** - Complete reference guide
- **[ACTION_FORMAT_COMPARISON.md](./ACTION_FORMAT_COMPARISON.md)** - Visual comparison: 36D vs 14D
- **[MODIFICATION_PINPOINT.md](./MODIFICATION_PINPOINT.md)** - Exact code locations

## What's the 14D Format?

The action expert now outputs **14 dimensions** instead of 36:

```
┌─────────────────────────────────────────┐
│  Left Hand 6DOF (6 dims)                │
│    - Position XYZ (3)                   │
│    - Rotation RPY (3)                   │
├─────────────────────────────────────────┤
│  Right Hand 6DOF (6 dims)               │
│    - Position XYZ (3)                   │
│    - Rotation RPY (3)                   │
├─────────────────────────────────────────┤
│  Head Camera Height (1 dim)             │
├─────────────────────────────────────────┤
│  Discrete Token (1 dim)                 │
└─────────────────────────────────────────┘
         Total: 14 dimensions
```

## Quick Start

```bash
# Train with 14D format
bash scripts/train/psi0/posttrain-he-psi0-14d.sh

# Or use original 36D format
bash scripts/train/psi0/posttrain-he-psi0.sh
```

## Document Overview

### MODIFICATION_SUMMARY.md
**Purpose**: Quick reference for developers  
**Contains**:
- What changed
- How to use
- Key training arguments
- Action format table

**Read this if**: You need a quick reminder of how to use the 14D format

### ACTION_FORMAT_14D_README.md
**Purpose**: Comprehensive guide  
**Contains**:
- Detailed architecture explanation
- Data flow diagrams
- API documentation
- Testing instructions
- Troubleshooting guide

**Read this if**: You're implementing new features or debugging issues

### ACTION_FORMAT_COMPARISON.md
**Purpose**: Visual comparison between formats  
**Contains**:
- Side-by-side comparison of 36D vs 14D
- Data flow diagrams
- Configuration differences
- Benefits and trade-offs

**Read this if**: You want to understand the differences between formats

### MODIFICATION_PINPOINT.md
**Purpose**: Exact code locations  
**Contains**:
- Line-by-line changes
- File paths and line numbers
- Code snippets
- Verification checklist

**Read this if**: You need to review or modify the implementation

### TRAINING_VERIFICATION.md
**Purpose**: Test results and proof of functionality  
**Contains**:
- Training logs
- Loss curves
- Issues encountered and fixed
- Performance metrics

**Read this if**: You want confirmation that it works or are debugging training issues

## File Modifications Summary

| File | Status | Description |
|------|--------|-------------|
| `src/psi/config/transform.py` | Modified | Added 14D extraction logic |
| `assets/stats/he_raw_rel_stats_14d.json` | New | 14D normalization stats |
| `scripts/train/psi0/posttrain-he-psi0-14d.sh` | New | 14D training script |

## Key Features

✅ **Backward Compatible** - Original 36D format still works  
✅ **No Model Changes** - Architecture adapts automatically  
✅ **Verified Working** - Successfully trained with 14D actions  
✅ **Well Documented** - 5 comprehensive documentation files  

## Testing Status

- ✅ Data loading with 14D format
- ✅ Model initialization with 14D actions
- ✅ Training runs successfully
- ✅ Loss converges properly
- ✅ W&B logging works

## Common Tasks

### Switch between formats

**Use 14D (hands-only)**:
```bash
bash scripts/train/psi0/posttrain-he-psi0-14d.sh
```

**Use 36D (full humanoid)**:
```bash
bash scripts/train/psi0/posttrain-he-psi0.sh
```

### Check action dimensions in code

```python
# In transform
action_format="hands_only"  # 14D
action_format="full"        # 36D (default)

# In training args
--model.action-dim=14       # 14D
--model.action-dim=36       # 36D
```

## Support

If you encounter issues:

1. Check **TRAINING_VERIFICATION.md** for known issues and solutions
2. Review **MODIFICATION_PINPOINT.md** to verify your changes
3. Consult **ACTION_FORMAT_14D_README.md** for detailed troubleshooting

## Changelog

**2026-04-02**: Initial implementation
- Created 14D action format
- Added extraction methods for wrist poses, head height, discrete token
- Fixed state dimension consistency
- Verified training works
- Created comprehensive documentation

---

**Author**: GitHub Copilot CLI  
**Date**: 2026-04-02  
**Version**: 1.0
