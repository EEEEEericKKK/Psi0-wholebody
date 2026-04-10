# Psi0 Wholebody VLA Documentation

This directory contains documentation for various modifications and features of the Psi0 Wholebody Vision-Language-Action model.

## Directory Structure

```
docs/
└── action_format_14d/          # 14D Action Format Modification
    ├── README.md               # Navigation guide
    ├── MODIFICATION_SUMMARY.md # Quick reference
    ├── ACTION_FORMAT_14D_README.md # Complete guide
    ├── ACTION_FORMAT_COMPARISON.md # 36D vs 14D comparison
    ├── MODIFICATION_PINPOINT.md    # Code locations
    └── TRAINING_VERIFICATION.md    # Test results
```

## Available Documentation

### Action Format 14D (Latest)
**Location**: [`action_format_14d/`](./action_format_14d/)

Comprehensive documentation for the 14D action format modification that changes the action expert output from 36D (full humanoid control) to 14D (2 hands 6DOF + head height + discrete token).

**Quick Start**: 
```bash
# Train with 14D format
bash scripts/train/psi0/posttrain-he-psi0-14d.sh
```

**Read More**: [action_format_14d/README.md](./action_format_14d/README.md)

## Contributing Documentation

When adding new features or modifications:

1. Create a new directory under `docs/` (e.g., `docs/feature_name/`)
2. Add a README.md in that directory explaining the feature
3. Include relevant documentation files
4. Update this main README with a link to your documentation

## Documentation Standards

Good documentation should include:

- ✅ **Summary**: What changed and why
- ✅ **Usage Guide**: How to use the feature
- ✅ **Code Locations**: Where to find the implementation
- ✅ **Examples**: Concrete usage examples
- ✅ **Testing**: How to verify it works
- ✅ **Troubleshooting**: Common issues and solutions

## Getting Help

For questions about:
- **14D Action Format**: See [action_format_14d/README.md](./action_format_14d/README.md)
- **General Psi0 Usage**: See main project README.md
- **Training Issues**: Check relevant documentation in this directory

---

Last Updated: 2026-04-02
