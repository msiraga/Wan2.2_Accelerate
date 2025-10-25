# Modifications to Original Wan 2.2 Repository

This document tracks changes made to the original Wan 2.2 codebase for T2V optimization.

## Modified Files

### 1. `wan/__init__.py`
**Purpose:** Make S2V, I2V, and Animate imports optional

**Change:** Wrapped optional imports in try/except blocks to gracefully handle missing dependencies

**Original:**
```python
from .image2video import WanI2V
from .speech2video import WanS2V
from .text2video import WanT2V
from .textimage2video import WanTI2V
from .animate import WanAnimate
```

**Modified:**
```python
# Core T2V imports (always available)
from .text2video import WanT2V

# Optional imports (require additional dependencies)
try:
    from .image2video import WanI2V
except ImportError:
    WanI2V = None

try:
    from .speech2video import WanS2V
except ImportError:
    WanS2V = None

try:
    from .textimage2video import WanTI2V
except ImportError:
    WanTI2V = None

try:
    from .animate import WanAnimate
except ImportError:
    WanAnimate = None
```

**Why:** Allows T2V to work without installing S2V dependencies (librosa, decord, soundfile, opencv-python)

**Impact:** 
- ✅ T2V works without optional dependencies
- ✅ Smaller installation footprint
- ✅ Tests pass without S2V libraries
- ⚠️ S2V/I2V/Animate features require their dependencies

---

## Added Files (Optimizations)

These are NEW files, not modifications:

1. **`wan/text2video_optimized.py`** - Optimized T2V pipeline class
2. **`generate_optimized.py`** - CLI script using optimized pipeline
3. **`test_quick.py`** - Quick validation script
4. **`test_with_checkpoint.py`** - End-to-end test with checkpoints
5. **`tests/test_optimized_t2v.py`** - Unit tests for optimizations
6. **Various documentation files** - Guides and instructions

---

## Updating from Upstream

If you need to pull updates from the original Wan 2.2 repo:

```bash
# The only conflict will be wan/__init__.py
# Simply re-apply the try/except wrapper to new imports
```

**Steps:**
1. Pull upstream changes: `git pull upstream main`
2. If `wan/__init__.py` conflicts, re-apply the try/except blocks
3. All other optimizations are in separate files (no conflicts)

---

## Rolling Back Changes

To restore original behavior:

```bash
# Revert wan/__init__.py
git checkout origin/main -- wan/__init__.py

# Install all dependencies
pip install decord librosa soundfile opencv-python
```

---

## Summary

**Modified:** 1 file (`wan/__init__.py`)  
**Added:** ~15 files (optimizations, tests, docs)  
**Impact:** Minimal, graceful dependency handling  
**Reversible:** Yes, easily

This modification makes the codebase more flexible and production-ready for T2V-focused deployments.

