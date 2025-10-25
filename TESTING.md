# 🧪 Testing Guide for Wan 2.2 T2V Optimizations

Complete testing suite to validate everything works before using H200.

---

## 📊 Test Levels

### Level 1: Quick Smoke Test (2 minutes, NO checkpoint needed)
Tests code structure and dependencies

### Level 2: Unit Tests (5 minutes, NO checkpoint needed)  
Tests individual components with mocks

### Level 3: Checkpoint Tests (10 minutes, checkpoint required)
End-to-end validation with real model

### Level 4: Full Benchmark (30 minutes, checkpoint required)
Performance comparison original vs optimized

---

## 🚀 Quick Start

### Step 1: Quick Smoke Test (Start Here!)

**No checkpoint needed, runs in 2 minutes:**

```bash
python test_quick.py
```

**What it tests:**
- ✓ PyTorch & CUDA available
- ✓ Imports work correctly
- ✓ Dependencies installed
- ✓ TF32 support
- ✓ Batched CFG logic
- ✓ torch.compile available
- ✓ Configuration loading

**Expected output:**
```
WAN 2.2 T2V - QUICK SMOKE TEST
Test 1: PyTorch & CUDA .................. PASS ✓
Test 2: Import Optimized Pipeline ....... PASS ✓
Test 3: Generate Script Validation ...... PASS ✓
Test 4: Required Dependencies ........... PASS ✓
Test 5: TF32 Support .................... PASS ✓
Test 6: Batched CFG Tensor Operations ... PASS ✓
Test 7: torch.compile Support ........... PASS ✓
Test 8: Configuration Loading ........... PASS ✓
Test 9: Class Instantiation (Mock) ...... PASS ✓

✓ All quick tests passed!
```

---

### Step 2: Unit Tests with Pytest (Optional)

**Requires pytest, no checkpoint needed:**

```bash
# Install pytest if needed
pip install pytest pytest-mock

# Run unit tests
pytest tests/test_optimized_t2v.py -v
```

**What it tests:**
- Module imports
- Class initialization  
- Optimization flags
- Batched CFG tensor operations
- Script structure

**Expected output:**
```
tests/test_optimized_t2v.py::TestOptimizedImports::test_import_optimized_pipeline PASSED
tests/test_optimized_t2v.py::TestOptimizedImports::test_import_generate_script PASSED
tests/test_optimized_t2v.py::TestOptimizedClass::test_tf32_enabled PASSED
tests/test_optimized_t2v.py::TestOptimizedClass::test_compile_mode_options PASSED
tests/test_optimized_t2v.py::TestBatchedCFG::test_batched_cfg_tensor_shapes PASSED
...
```

---

### Step 3: Checkpoint Test (Before H200)

**Download checkpoint first:**

```bash
# Download checkpoint (~56GB, 10-30 min)
pip install "huggingface_hub[cli,hf_transfer]"
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download \
    Wan-AI/Wan2.2-T2V-A14B \
    --local-dir ./checkpoints/Wan2.2-T2V-A14B
```

**Then run checkpoint test:**

```bash
# Quick test (5 frames, 2 steps, ~2 min)
python test_with_checkpoint.py \
    --ckpt_dir ./checkpoints/Wan2.2-T2V-A14B \
    --quick

# Or normal test (17 frames, 10 steps, ~5 min)
python test_with_checkpoint.py \
    --ckpt_dir ./checkpoints/Wan2.2-T2V-A14B
```

**What it tests:**
- ✓ Checkpoint structure
- ✓ Pipeline loading
- ✓ Video generation
- ✓ Optimizations active
- ✓ Batched CFG working
- ✓ Memory usage

**Expected output:**
```
WAN 2.2 T2V - CHECKPOINT TEST
Test 1: Checkpoint Structure ............ PASS ✓
Test 2: Load Optimized Pipeline ......... PASS ✓
Test 3: Video Generation ................ PASS ✓
  ✓ Time: 45.2s
  ✓ Peak VRAM: 28.3 GB
  ✓ Video shape: (3, 17, 720, 1280)
Test 4: Verify Optimizations ............ PASS ✓
Test 5: Batched CFG Validation .......... PASS ✓

🎉 Ready for production use on H200!
```

---

### Step 4: Full Benchmark (Final Validation)

**Compare original vs optimized:**

```bash
python benchmark_t2v.py \
    --ckpt_dir ./checkpoints/Wan2.2-T2V-A14B \
    --mode both \
    --num_runs 3 \
    --frame_num 49 \
    --no-offload
```

**What it tests:**
- Original implementation performance
- Optimized implementation performance  
- Speedup calculation
- Memory comparison

**Expected output:**
```
COMPARISON RESULTS
==================
Original Implementation:
  Average Time: 180.5s
  Peak Memory:  62.4 GB

Optimized Implementation:
  Average Time: 58.2s
  Peak Memory:  78.1 GB

Performance Gain:
  Speedup:      3.10x
  Time Saved:   122.3s (67.7%)
```

---

## 📋 Test Checklist

Before investing in H200:

- [ ] **Step 1 Complete:** Quick smoke test passed
- [ ] **Step 2 Complete:** Unit tests passed (optional)
- [ ] **Step 3 Complete:** Checkpoint test passed
- [ ] **Step 4 Complete:** Benchmark shows 2.5x+ speedup

✅ **All green?** You're ready for H200!

---

## 🔍 Detailed Test Descriptions

### Test Files

```
tests/
├── test_optimized_t2v.py    # Pytest unit tests
test_quick.py                 # Quick smoke test
test_with_checkpoint.py       # Checkpoint validation
benchmark_t2v.py              # Performance benchmark
```

### Quick Smoke Test (`test_quick.py`)

**Purpose:** Verify code structure without checkpoint  
**Time:** 2 minutes  
**GPU:** Not required  

**Tests:**
1. PyTorch and CUDA availability
2. Module imports work
3. Script structure valid
4. Dependencies installed
5. TF32 can be enabled
6. Batched CFG tensor logic
7. torch.compile available
8. Config loading works
9. Class instantiation (mocked)

**Run:**
```bash
python test_quick.py
```

---

### Unit Tests (`tests/test_optimized_t2v.py`)

**Purpose:** Test individual components  
**Time:** 5 minutes  
**GPU:** Optional  

**Test Classes:**
- `TestOptimizedImports` - Import validation
- `TestOptimizedClass` - Class initialization
- `TestBatchedCFG` - CFG tensor operations
- `TestGenerateScript` - Script validation
- `TestWithCheckpoint` - Checkpoint tests (optional)

**Run:**
```bash
# All tests
pytest tests/test_optimized_t2v.py -v

# Specific test class
pytest tests/test_optimized_t2v.py::TestBatchedCFG -v

# With checkpoint tests
RUN_CHECKPOINT_TESTS=1 CKPT_DIR=/path pytest tests/test_optimized_t2v.py -v
```

---

### Checkpoint Test (`test_with_checkpoint.py`)

**Purpose:** End-to-end validation  
**Time:** 5-10 minutes  
**GPU:** Required  
**Checkpoint:** Required  

**Tests:**
1. Checkpoint file structure
2. Pipeline loading (1-2 min)
3. Video generation works
4. Optimizations are active
5. Batched CFG faster than unbatched

**Run:**
```bash
# Quick (5 frames, 2 steps)
python test_with_checkpoint.py \
    --ckpt_dir ./checkpoints/Wan2.2-T2V-A14B \
    --quick

# Normal (17 frames, 10 steps)
python test_with_checkpoint.py \
    --ckpt_dir ./checkpoints/Wan2.2-T2V-A14B
```

---

### Benchmark (`benchmark_t2v.py`)

**Purpose:** Performance comparison  
**Time:** 20-30 minutes  
**GPU:** Required  
**Checkpoint:** Required  

**Compares:**
- Original vs Optimized
- Multiple runs (averaging)
- Time and memory
- Speedup calculation

**Run:**
```bash
# Quick benchmark (49 frames)
python benchmark_t2v.py \
    --ckpt_dir ./checkpoints/Wan2.2-T2V-A14B \
    --mode both \
    --num_runs 3 \
    --frame_num 49

# Full benchmark (81 frames)
python benchmark_t2v.py \
    --ckpt_dir ./checkpoints/Wan2.2-T2V-A14B \
    --mode both \
    --num_runs 3 \
    --frame_num 81 \
    --no-offload
```

---

## ⚠️ Common Issues

### Issue: "CUDA not available"
**Solution:**
```bash
# Check CUDA
nvidia-smi

# Check PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall PyTorch with CUDA
pip install torch>=2.4.0 --index-url https://download.pytorch.org/whl/cu121
```

### Issue: "Cannot import WanT2VOptimized"
**Solution:**
```bash
# Make sure you're in project root
cd /path/to/Wan2.2

# Check file exists
ls wan/text2video_optimized.py

# Run test
python test_quick.py
```

### Issue: "Checkpoint not found"
**Solution:**
```bash
# Verify checkpoint path
ls -la ./checkpoints/Wan2.2-T2V-A14B/

# Should contain:
# - low_noise_model/
# - high_noise_model/
# - models_t5_umt5-xxl-enc-bf16.pth
# - Wan2.1_VAE.pth

# Re-download if missing
huggingface-cli download Wan-AI/Wan2.2-T2V-A14B --local-dir ./checkpoints/Wan2.2-T2V-A14B
```

### Issue: "Out of memory" in tests
**Solution:**
```bash
# Use quick mode
python test_with_checkpoint.py --ckpt_dir /path --quick

# Or enable offloading in benchmark
python benchmark_t2v.py --ckpt_dir /path --mode optimized --offload_model True
```

---

## 📊 Expected Results Summary

### Local Development (Before H200)

| Test | Time | Pass Criteria |
|------|------|---------------|
| Quick Smoke | 2 min | All 9 tests pass |
| Unit Tests | 5 min | All pytest green |
| Checkpoint Test | 5-10 min | Generation successful |
| Benchmark | 20-30 min | 2.5x+ speedup |

### On H200 (After Deployment)

| Metric | Expected |
|--------|----------|
| Load time | 1-2 min |
| First gen (81f) | ~120s (with compile) |
| Subsequent (81f) | ~85s |
| Speedup | 3.3-3.7x |
| Peak VRAM | ~78GB / 141GB |

---

## ✅ Final Validation Checklist

Before deploying to H200:

### Code Validation
- [ ] `python test_quick.py` - All pass
- [ ] `pytest tests/test_optimized_t2v.py -v` - All green (optional)

### Checkpoint Validation  
- [ ] Downloaded Wan2.2-T2V-A14B (~56GB)
- [ ] `python test_with_checkpoint.py --ckpt_dir /path` - Pass
- [ ] Generated video looks correct

### Performance Validation
- [ ] `python benchmark_t2v.py --ckpt_dir /path --mode both` - Pass
- [ ] Speedup > 2.5x confirmed
- [ ] Memory usage acceptable

### Ready for H200! ✅
- [ ] All tests passed
- [ ] Understand the commands
- [ ] Read `RUNPOD_H200_SETUP.md`

---

## 🎯 Recommended Testing Sequence

**Timeline: ~1 hour total**

```
1. Quick smoke test           (2 min)   ← START HERE
   └─ python test_quick.py

2. Download checkpoint        (15-30 min)
   └─ huggingface-cli download ...

3. Checkpoint test            (5 min)
   └─ python test_with_checkpoint.py --quick

4. Short benchmark            (20 min)
   └─ python benchmark_t2v.py --frame_num 49

✅ All pass? Ready for H200!
```

---

## 📞 Need Help?

If any test fails:

1. Check error message carefully
2. Look in "Common Issues" section above
3. Verify you're in project root directory
4. Check CUDA is available: `nvidia-smi`
5. Check all dependencies: `pip install -r requirements.txt`

---

*Last updated: 2025-10-25*  
*Tested on: PyTorch 2.4+, CUDA 12.1+, Various GPUs*

