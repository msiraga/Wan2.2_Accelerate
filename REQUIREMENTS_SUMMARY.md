# 📦 Requirements Summary

## Changes Made to `requirements.txt`

### ✅ **Added (Missing Dependencies)**
- `pillow>=9.0.0` - Required for PIL/Image operations
- `einops` - Required for tensor operations in VAE
- `regex` - Required for tokenizer text processing
- `safetensors>=0.4.0` - Required for loading model checkpoints

### ❌ **Removed (Not Needed for T2V)**
- `torchaudio` - Only needed for S2V (Speech-to-Video)
  - Moved to `requirements_optional.txt`
- `dashscope` - Only for optional prompt extension API
  - Moved to `requirements_optional.txt`
- `imageio-ffmpeg` - Redundant (already in `imageio[ffmpeg]`)

### ✨ **Kept & Organized**
All other dependencies are required and organized by category:
- Core PyTorch
- Diffusion & Transformers
- Flash Attention (critical!)
- Image & Video Processing
- Text Processing
- Utilities
- Model Loading

---

## 📋 Installation Instructions

### Basic T2V (Recommended)
```bash
pip install -r requirements.txt
```

This installs everything needed for **optimized T2V generation**.

### Optional Features
```bash
# For prompt extension, S2V, testing, etc.
pip install -r requirements_optional.txt
```

### Specific Optional Features
```bash
# Just prompt extension
pip install dashscope

# Just S2V
pip install torchaudio

# Just testing
pip install pytest pytest-mock
```

---

## 🔍 Dependency Count

### Before: 16 packages
- Included unnecessary dependencies
- Missing critical dependencies
- Had redundant entries

### After: 18 packages (main) + optional file
- ✅ All required dependencies included
- ✅ No unnecessary dependencies in main file
- ✅ Better organized with comments
- ✅ Optional dependencies separated

---

## 💾 Estimated Install Size

| Category | Size |
|----------|------|
| PyTorch (torch, torchvision) | ~2.5 GB |
| Transformers & Diffusers | ~500 MB |
| Flash Attention | ~100 MB |
| Other packages | ~300 MB |
| **Total** | **~3.4 GB** |

*Note: Model checkpoints (~56GB) are separate*

---

## ⚡ Installation Tips

### Fast Installation
```bash
# Use faster package index for China
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# Or use pip with cache
pip install --cache-dir ~/.cache/pip -r requirements.txt
```

### If flash_attn Fails
```bash
# Install other packages first
pip install -r requirements.txt --no-deps
pip install torch torchvision diffusers transformers

# Then install flash_attn separately
pip install flash_attn --no-build-isolation
```

### GPU-Specific PyTorch
```bash
# For CUDA 12.1
pip install torch>=2.4.0 torchvision>=0.19.0 --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8
pip install torch>=2.4.0 torchvision>=0.19.0 --index-url https://download.pytorch.org/whl/cu118
```

---

## ✅ Verification

After installation, verify everything works:

```bash
# Quick check
python -c "import torch; import diffusers; import transformers; import einops; import regex; print('✓ All imports successful')"

# Full check
python test_quick.py
```

---

## 🐛 Known Issues

### Issue: `regex` module conflicts with `re`
**Solution**: The code uses `import regex as re`, which is correct. No action needed.

### Issue: Flash Attention compilation fails
**Solution**: 
```bash
pip install --upgrade pip setuptools wheel
pip install flash_attn --no-build-isolation
```

### Issue: `pillow` version conflicts
**Solution**: Most packages are compatible with Pillow 9+. If issues:
```bash
pip install pillow==10.0.0
```

---

## 📊 Dependency Tree (Simplified)

```
torch>=2.4.0
├── numpy
└── pillow

diffusers>=0.31.0
├── transformers
│   ├── tokenizers
│   ├── regex
│   └── safetensors
└── accelerate

flash_attn
├── torch
└── einops

imageio[ffmpeg]
├── numpy
└── pillow

opencv-python
└── numpy
```

---

## 🎯 Summary

The updated `requirements.txt`:
- ✅ Contains ONLY what's needed for T2V
- ✅ Includes ALL required dependencies
- ✅ Well organized with comments
- ✅ Separates optional dependencies
- ✅ Ready for H200 deployment

**Result**: Clean, complete, and optimized dependency list! 🚀

