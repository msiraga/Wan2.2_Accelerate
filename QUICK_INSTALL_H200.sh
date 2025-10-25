#!/bin/bash
# Quick installation script for Wan 2.2 T2V on H200 RunPod
# Run this after SSH'ing into your H200 instance

set -e  # Exit on error

echo "=========================================="
echo "Wan 2.2 T2V - H200 Quick Setup"
echo "=========================================="
echo ""

# ========================================
# STEP 1: Check Environment
# ========================================
echo "→ Checking environment..."

# Check Python
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1-2)
echo "  Python version: $PYTHON_VERSION"

if [ "$PYTHON_VERSION" != "3.11" ] && [ "$PYTHON_VERSION" != "3.10" ]; then
    echo "  ⚠️  Warning: Python 3.10 or 3.11 recommended"
fi

# Check GPU
echo ""
echo "→ Checking GPU..."
nvidia-smi --query-gpu=gpu_name,memory.total --format=csv,noheader | head -n1

# ========================================
# STEP 2: Update System Packages
# ========================================
echo ""
echo "→ Updating system packages..."
apt-get update > /dev/null 2>&1 || true
apt-get install -y git wget build-essential > /dev/null 2>&1 || true

# ========================================
# STEP 3: Install PyTorch (if needed)
# ========================================
echo ""
echo "→ Checking PyTorch..."

if python3 -c "import torch" 2>/dev/null; then
    TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null)
    CUDA_VERSION=$(python3 -c "import torch; print(torch.version.cuda)" 2>/dev/null)
    echo "  PyTorch: $TORCH_VERSION (CUDA $CUDA_VERSION)"
    
    # Check if version is adequate
    if [[ "$TORCH_VERSION" < "2.4" ]]; then
        echo "  ⚠️  PyTorch version too old, upgrading..."
        pip3 install --upgrade torch>=2.4.0 torchvision torchaudio
    else
        echo "  ✓ PyTorch version OK"
    fi
else
    echo "  Installing PyTorch 2.4+ with CUDA support..."
    pip3 install torch>=2.4.0 torchvision torchaudio
fi

# ========================================
# STEP 4: Install Core Dependencies
# ========================================
echo ""
echo "→ Installing core dependencies..."

# Install requirements line by line to catch errors
pip3 install transformers>=4.45.1 || echo "⚠️  transformers failed"
pip3 install diffusers>=0.31.0 || echo "⚠️  diffusers failed"
pip3 install accelerate>=1.1.1 || echo "⚠️  accelerate failed"
pip3 install pillow || echo "⚠️  pillow failed"
pip3 install einops || echo "⚠️  einops failed"
pip3 install sentencepiece || echo "⚠️  sentencepiece failed"
pip3 install protobuf || echo "⚠️  protobuf failed"
pip3 install regex || echo "⚠️  regex failed"
pip3 install safetensors || echo "⚠️  safetensors failed"
pip3 install "numpy<2" || echo "⚠️  numpy failed"
pip3 install psutil || echo "⚠️  psutil failed"

# ========================================
# STEP 5: Install Flash Attention
# ========================================
echo ""
echo "→ Installing flash-attn (this takes 3-5 minutes)..."
echo "  (Building from source for H200 optimization)"

pip3 install flash-attn --no-build-isolation || {
    echo "⚠️  flash-attn installation failed"
    echo "  → Trying alternative installation..."
    pip3 install flash-attn || echo "✗ flash-attn completely failed (not critical)"
}

# ========================================
# STEP 6: Verify Installation
# ========================================
echo ""
echo "=========================================="
echo "Verification"
echo "=========================================="
echo ""

# Test imports
python3 -c "
import sys
import torch
import transformers
import diffusers
import einops

print('✓ PyTorch:', torch.__version__)
print('✓ CUDA available:', torch.cuda.is_available())
print('✓ GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')
print('✓ Transformers:', transformers.__version__)
print('✓ Diffusers:', diffusers.__version__)

try:
    import flash_attn
    print('✓ Flash Attention:', flash_attn.__version__)
except ImportError:
    print('⚠️  Flash Attention: Not installed (optional but recommended)')

print('')
print('='*50)
print('✓ Installation complete!')
print('='*50)
"

echo ""
echo "=========================================="
echo "Next Steps:"
echo "=========================================="
echo ""
echo "1. Download checkpoint:"
echo "   python3 generate_optimized.py --download-checkpoint"
echo ""
echo "2. Run quick test:"
echo "   python3 test_quick.py"
echo ""
echo "3. Generate video:"
echo "   python3 generate_optimized.py --prompt 'A cat playing piano' --compile --tf32"
echo ""

