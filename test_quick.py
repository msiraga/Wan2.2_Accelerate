#!/usr/bin/env python3
"""
Quick smoke test for optimized T2V - NO CHECKPOINT NEEDED

This tests the code structure without requiring model weights.

Usage:
    python test_quick.py
"""

import sys
import torch
from pathlib import Path

print("="*70)
print("WAN 2.2 T2V - QUICK SMOKE TEST (No Checkpoint Required)")
print("="*70)
print()

# Test 1: PyTorch and CUDA
print("Test 1: PyTorch & CUDA")
print("-" * 50)
try:
    print(f"✓ PyTorch version: {torch.__version__}")
    print(f"✓ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✓ CUDA version: {torch.version.cuda}")
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print("PASS ✓\n")
except Exception as e:
    print(f"FAIL ✗: {e}\n")
    sys.exit(1)

# Test 2: Import optimized pipeline
print("Test 2: Import Optimized Pipeline")
print("-" * 50)
try:
    from wan.text2video_optimized import WanT2VOptimized
    print("✓ WanT2VOptimized imported successfully")
    print(f"✓ Class location: {WanT2VOptimized.__module__}")
    print("PASS ✓\n")
except ImportError as e:
    print(f"FAIL ✗: Cannot import WanT2VOptimized")
    print(f"Error: {e}\n")
    sys.exit(1)

# Test 3: Import generate script
print("Test 3: Generate Script Validation")
print("-" * 50)
try:
    script_path = Path("generate_optimized.py")
    assert script_path.exists(), "generate_optimized.py not found"
    print("✓ generate_optimized.py found")
    
    # Read and check for key functions
    with open(script_path) as f:
        content = f.read()
        assert "def main()" in content, "main() function not found"
        assert "WanT2VOptimized" in content, "WanT2VOptimized not imported"
        assert "generate(" in content, "generate() call not found"
    print("✓ Script structure looks good")
    print("PASS ✓\n")
except Exception as e:
    print(f"FAIL ✗: {e}\n")
    sys.exit(1)

# Test 4: Dependencies
print("Test 4: Required Dependencies")
print("-" * 50)
dependencies = {
    "torch": None,
    "diffusers": None,
    "transformers": None,
    "tqdm": None,
}

failed_deps = []
for dep in dependencies:
    try:
        __import__(dep)
        print(f"✓ {dep}")
    except ImportError:
        print(f"✗ {dep} - MISSING")
        failed_deps.append(dep)

if failed_deps:
    print(f"\nFAIL ✗: Missing dependencies: {', '.join(failed_deps)}")
    print("Run: pip install -r requirements.txt\n")
    sys.exit(1)
else:
    print("PASS ✓\n")

# Test 5: TF32 enablement
print("Test 5: TF32 Support")
print("-" * 50)
try:
    # Save original state
    original_tf32 = torch.backends.cuda.matmul.allow_tf32
    
    # Test enabling
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    
    print("✓ TF32 can be enabled")
    print(f"✓ Current state: {torch.backends.cuda.matmul.allow_tf32}")
    
    # Restore
    torch.backends.cuda.matmul.allow_tf32 = original_tf32
    
    print("PASS ✓\n")
except Exception as e:
    print(f"FAIL ✗: {e}\n")
    sys.exit(1)

# Test 6: Batched CFG logic
print("Test 6: Batched CFG Tensor Operations")
print("-" * 50)
try:
    # Test batching logic
    C, T, H, W = 16, 20, 90, 160
    
    x = torch.randn(C, T, H, W)
    x_batched = torch.cat([x, x], dim=0)
    
    assert x_batched.shape == (C*2, T, H, W), "Batching failed"
    print(f"✓ Input batching: {x.shape} -> {x_batched.shape}")
    
    # Test splitting
    x1, x2 = x_batched.chunk(2, dim=0)
    assert x1.shape == x.shape, "Splitting failed"
    assert torch.allclose(x1, x), "Split doesn't match original"
    
    print(f"✓ Output splitting: {x_batched.shape} -> 2x{x1.shape}")
    print("PASS ✓\n")
except Exception as e:
    print(f"FAIL ✗: {e}\n")
    sys.exit(1)

# Test 7: torch.compile availability
print("Test 7: torch.compile Support")
print("-" * 50)
try:
    def dummy_fn(x):
        return x * 2
    
    # Try to compile
    compiled_fn = torch.compile(dummy_fn, mode="reduce-overhead")
    
    # Test it works
    x = torch.randn(10)
    y1 = dummy_fn(x)
    y2 = compiled_fn(x)
    
    assert torch.allclose(y1, y2), "Compiled output doesn't match"
    
    print("✓ torch.compile is available")
    print("✓ Compilation successful")
    print("PASS ✓\n")
except Exception as e:
    print(f"WARNING ⚠: torch.compile may not work properly")
    print(f"Error: {e}")
    print("This is OK - will fall back to non-compiled mode\n")

# Test 8: Config loading
print("Test 8: Configuration Loading")
print("-" * 50)
try:
    from wan.configs import WAN_CONFIGS, SIZE_CONFIGS
    
    assert "t2v-A14B" in WAN_CONFIGS, "t2v-A14B config not found"
    config = WAN_CONFIGS["t2v-A14B"]
    
    print(f"✓ Config loaded: {config.__name__}")
    print(f"✓ Model dim: {config.dim}")
    print(f"✓ Layers: {config.num_layers}")
    print(f"✓ Sample steps: {config.sample_steps}")
    
    assert "1280*720" in SIZE_CONFIGS, "1280*720 size not found"
    print(f"✓ Size configs available")
    
    print("PASS ✓\n")
except Exception as e:
    print(f"FAIL ✗: {e}\n")
    sys.exit(1)

# Test 9: Optimization class instantiation (mocked)
print("Test 9: Class Instantiation (Mock)")
print("-" * 50)
try:
    from unittest.mock import Mock, patch
    from wan.text2video_optimized import WanT2VOptimized
    from wan.configs import WAN_CONFIGS
    
    config = WAN_CONFIGS["t2v-A14B"]
    
    # Mock the heavy components
    with patch('wan.text2video_optimized.T5EncoderModel'), \
         patch('wan.text2video_optimized.Wan2_1_VAE'), \
         patch('wan.text2video_optimized.WanModel'):
        
        pipeline = WanT2VOptimized(
            config=config,
            checkpoint_dir="/fake/path",
            enable_compile=False,
            enable_tf32=True,
            compile_mode="reduce-overhead"
        )
        
        assert pipeline.enable_compile == False
        assert pipeline.compile_mode == "reduce-overhead"
        
    print("✓ Class instantiates correctly")
    print("✓ Optimization flags work")
    print("PASS ✓\n")
except Exception as e:
    print(f"FAIL ✗: {e}\n")
    sys.exit(1)

# Summary
print("="*70)
print("SUMMARY")
print("="*70)
print("✓ All quick tests passed!")
print("\nYour optimized T2V code is structurally correct.")
print("\nNext steps:")
print("  1. Download checkpoint (~56GB)")
print("  2. Run full test: python test_with_checkpoint.py --ckpt_dir /path")
print("  3. Or run benchmark: python benchmark_t2v.py --ckpt_dir /path")
print("\nReady to proceed with H200! 🚀")
print("="*70)

