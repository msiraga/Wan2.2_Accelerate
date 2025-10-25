# 🚀 Installation Instructions for Wan 2.2 T2V Optimized

Complete guide to install dependencies correctly.

---

## ⚡ Quick Install (Recommended)

### Method 1: Two-Step Installation (Most Reliable)

**Step 1: Install PyTorch first**
```bash
# For H200 (auto-detect CUDA - recommended)
pip install torch>=2.4.0 torchvision>=0.19.0

# Or explicitly for CUDA 12.4 (H200 optimized)
pip install torch>=2.4.0 torchvision>=0.19.0 --index-url https://download.pytorch.org/whl/cu124

# Or for CUDA 12.1 (most stable, works on all GPUs)
pip install torch>=2.4.0 torchvision>=0.19.0 --index-url https://download.pytorch.org/whl/cu121

# Or for CUDA 11.8 (older GPUs)
pip install torch>=2.4.0 torchvision>=0.19.0 --index-url https://download.pytorch.org/whl/cu118
```

**For H200:** CUDA 12.1+ recommended. Auto-detect usually picks the best version.

**Step 2: Install remaining dependencies**
```bash
pip install -r requirements.txt
```

If flash-attn fails, install it separately:
```bash
pip install flash-attn --no-build-isolation
```

---

## 🪟 Windows Users (You!)

Flash Attention can be tricky on Windows. Here's the **recommended approach**:

### Option A: Install Pre-built Wheel (Fastest)

```bash
# 1. Install PyTorch first
pip install torch>=2.4.0 torchvision>=0.19.0 --index-url https://download.pytorch.org/whl/cu121

# 2. Try pre-built flash-attn wheel
pip install flash-attn --no-build-isolation

# 3. Install other dependencies
pip install -r requirements.txt
```

### Option B: Without Flash Attention (Still Works!)

If flash-attn installation keeps failing, **you can skip it**:

```bash
# Install everything except flash-attn
pip install torch>=2.4.0 torchvision>=0.19.0
pip install diffusers transformers tokenizers accelerate
pip install opencv-python imageio pillow einops
pip install ftfy regex easydict tqdm safetensors
```

**Note:** The code will automatically fall back to PyTorch's built-in attention if flash-attn is unavailable. You'll still get optimization benefits, just slightly slower attention operations.

---

## 🐧 Linux Users

### Standard Installation

```bash
# Install in one go
pip install -r requirements.txt

# If flash-attn fails
pip install torch>=2.4.0 torchvision>=0.19.0
pip install flash-attn --no-build-isolation
pip install -r requirements.txt
```

### Using Conda (Alternative)

```bash
conda create -n wan22 python=3.10
conda activate wan22
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```

---

## 🍎 macOS Users

**Note:** Flash Attention requires NVIDIA GPU. On macOS (Apple Silicon or Intel):

```bash
# Install PyTorch for MPS (Apple Silicon)
pip install torch>=2.4.0 torchvision>=0.19.0

# Skip flash-attn, install other dependencies
pip install diffusers transformers tokenizers accelerate
pip install opencv-python imageio pillow einops
pip install ftfy regex easydict tqdm safetensors
```

The code will use PyTorch's native attention on macOS.

---

## 🔧 Troubleshooting Flash Attention

### Error: "ModuleNotFoundError: No module named 'torch'"

**Solution:** Install torch first!
```bash
pip install torch>=2.4.0 torchvision>=0.19.0
pip install flash-attn --no-build-isolation
```

### Error: "Microsoft Visual C++ 14.0 or greater is required" (Windows)

**Solution:** Install Visual Studio Build Tools
1. Download: https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022
2. Install "Desktop development with C++"
3. Restart terminal
4. Try again: `pip install flash-attn --no-build-isolation`

**Or:** Skip flash-attn (code still works without it!)

### Error: "nvcc not found" or "CUDA not available"

**Solution:** Install CUDA Toolkit
1. Download from: https://developer.nvidia.com/cuda-downloads
2. Or use pre-built PyTorch wheel (includes CUDA):
   ```bash
   pip install torch>=2.4.0 torchvision>=0.19.0 --index-url https://download.pytorch.org/whl/cu121
   ```

### Error: Build takes too long or hangs

**Solution:** Use pre-built wheel
```bash
pip install --upgrade pip
pip install flash-attn --no-build-isolation
```

Or skip it - the optimizations still work!

---

## ✅ Verify Installation

After installation, run:

```bash
# Quick check
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# Check flash-attn (optional)
python -c "try:
    import flash_attn
    print('✓ Flash Attention available')
except:
    print('⚠ Flash Attention not available (will use fallback)')"

# Full validation
python test_quick.py
```

---

## 📦 Installation Order Summary

**Correct Order:**
1. ✅ PyTorch (`torch`, `torchvision`)
2. ✅ Flash Attention (`flash-attn`) - optional but recommended
3. ✅ Everything else (`requirements.txt`)

**Why this order?**
- Flash Attention needs PyTorch to build
- Other packages are independent

---

## 🎯 Recommended Installation Paths

### For RunPod / Cloud GPU
```bash
cd /workspace/Wan2.2
pip install torch>=2.4.0 torchvision>=0.19.0 --index-url https://download.pytorch.org/whl/cu121
pip install flash-attn --no-build-isolation
pip install -r requirements.txt
```

### For Local Windows Development
```bash
# In your project directory
python -m venv .venv
.venv\Scripts\activate
pip install torch>=2.4.0 torchvision>=0.19.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
# Skip flash-attn if it fails - code will work without it!
```

### For Local Linux Development
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 🚨 Common Mistakes

### ❌ Don't Do This:
```bash
# Installing flash-attn before torch
pip install flash-attn  # Will fail!
pip install torch
```

### ✅ Do This Instead:
```bash
# Install torch first
pip install torch>=2.4.0 torchvision>=0.19.0
pip install flash-attn --no-build-isolation
```

---

## 💡 Pro Tips

### Tip 1: Use Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
```

### Tip 2: Upgrade pip First
```bash
python -m pip install --upgrade pip setuptools wheel
```

### Tip 3: Use --no-build-isolation for flash-attn
```bash
pip install flash-attn --no-build-isolation
```

### Tip 4: Check CUDA Version
```bash
nvidia-smi
# Look for "CUDA Version: XX.X"
```

Then install matching PyTorch:
- CUDA 12.1: `--index-url https://download.pytorch.org/whl/cu121`
- CUDA 11.8: `--index-url https://download.pytorch.org/whl/cu118`

---

## 🎉 Success Checklist

After installation, you should have:

- [x] PyTorch 2.4+ with CUDA support
- [x] All required packages from requirements.txt
- [x] Flash Attention (optional but recommended)
- [x] Test script passes: `python test_quick.py`

---

## 🆘 Still Having Issues?

### Quick Fixes:

1. **Update pip:**
   ```bash
   python -m pip install --upgrade pip
   ```

2. **Install without flash-attn:**
   ```bash
   # Edit requirements.txt, comment out flash-attn line
   # Or install manually skipping it
   pip install torch>=2.4.0 torchvision>=0.19.0
   pip install diffusers transformers tokenizers accelerate
   pip install opencv-python imageio pillow einops
   pip install ftfy regex easydict tqdm safetensors
   ```

3. **Use pre-built wheels:**
   - Visit: https://github.com/Dao-AILab/flash-attention/releases
   - Download .whl file for your platform
   - `pip install downloaded_wheel.whl`

4. **Skip flash-attn entirely:**
   - The code works without it!
   - You'll use PyTorch's native attention (slightly slower but functional)

---

## 📞 Need More Help?

1. Check `TESTING.md` for validation steps
2. Check `REQUIREMENTS_SUMMARY.md` for dependency details
3. Run `python test_quick.py` to diagnose issues
4. Check CUDA: `nvidia-smi`
5. Check Python: `python --version` (need 3.10+)

---

**Bottom Line:** If flash-attn won't install, **skip it**! The optimizations still work, and you'll still get significant speedup from batched CFG, TF32, and torch.compile. 🚀

