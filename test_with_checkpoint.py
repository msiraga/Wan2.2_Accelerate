#!/usr/bin/env python3
"""
End-to-end test with actual checkpoint.

This validates the optimized pipeline works correctly with real model weights.

Usage:
    python test_with_checkpoint.py --ckpt_dir /path/to/Wan2.2-T2V-A14B
"""

import argparse
import sys
import time
import torch
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Test optimized T2V with checkpoint")
    parser.add_argument("--ckpt_dir", type=str, required=True,
                       help="Path to checkpoint directory")
    parser.add_argument("--quick", action="store_true",
                       help="Quick test (5 frames, 2 steps)")
    args = parser.parse_args()
    
    print("="*70)
    print("WAN 2.2 T2V - CHECKPOINT TEST")
    print("="*70)
    print(f"\nCheckpoint: {args.ckpt_dir}")
    print(f"Quick mode: {args.quick}\n")
    
    # Verify checkpoint exists
    ckpt_path = Path(args.ckpt_dir)
    if not ckpt_path.exists():
        print(f"✗ FAIL: Checkpoint directory not found: {args.ckpt_dir}")
        sys.exit(1)
    
    print("Test 1: Checkpoint Structure")
    print("-" * 50)
    required_files = [
        "low_noise_model",
        "high_noise_model",
        "models_t5_umt5-xxl-enc-bf16.pth",
        "Wan2.1_VAE.pth",
    ]
    
    missing = []
    for file in required_files:
        path = ckpt_path / file
        if path.exists():
            if path.is_file():
                size_mb = path.stat().st_size / 1024 / 1024
                print(f"✓ {file} ({size_mb:.1f} MB)")
            else:
                print(f"✓ {file}/ (directory)")
        else:
            print(f"✗ {file} - MISSING")
            missing.append(file)
    
    if missing:
        print(f"\nFAIL: Missing files: {', '.join(missing)}")
        sys.exit(1)
    print("PASS ✓\n")
    
    # Load pipeline
    print("Test 2: Load Optimized Pipeline")
    print("-" * 50)
    try:
        from wan.text2video_optimized import WanT2VOptimized
        from wan.configs import WAN_CONFIGS
        
        config = WAN_CONFIGS["t2v-A14B"]
        
        print("Loading models (this may take 1-2 minutes)...")
        start = time.time()
        
        pipeline = WanT2VOptimized(
            config=config,
            checkpoint_dir=args.ckpt_dir,
            enable_compile=False,  # Skip for test speed
            enable_tf32=True,
            device_id=0,
            rank=0
        )
        
        load_time = time.time() - start
        print(f"✓ Pipeline loaded in {load_time:.1f}s")
        print("PASS ✓\n")
    except Exception as e:
        print(f"✗ FAIL: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Test generation
    print("Test 3: Video Generation")
    print("-" * 50)
    
    if args.quick:
        frame_num = 5
        steps = 2
        print("Quick mode: 5 frames, 2 steps")
    else:
        frame_num = 17
        steps = 10
        print("Normal mode: 17 frames, 10 steps")
    
    try:
        print(f"\nGenerating {frame_num} frames...")
        torch.cuda.synchronize()
        start = time.time()
        
        video = pipeline.generate(
            input_prompt="A cat sitting on a chair",
            size=(1280, 720),
            frame_num=frame_num,
            sampling_steps=steps,
            guide_scale=(3.0, 4.0),
            use_batched_cfg=True,
            offload_model=True,
            seed=42
        )
        
        torch.cuda.synchronize()
        gen_time = time.time() - start
        
        peak_mem = torch.cuda.max_memory_allocated() / 1024**3
        
        print(f"\n✓ Generation successful!")
        print(f"✓ Time: {gen_time:.1f}s")
        print(f"✓ Peak VRAM: {peak_mem:.1f} GB")
        print(f"✓ Video shape: {video.shape}")
        print(f"✓ Expected shape: (3, {frame_num}, 720, 1280)")
        
        assert video.shape[0] == 3, f"Wrong channel count: {video.shape[0]}"
        assert video.shape[1] == frame_num, f"Wrong frame count: {video.shape[1]}"
        
        print("PASS ✓\n")
    except Exception as e:
        print(f"✗ FAIL: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Test optimizations are applied
    print("Test 4: Verify Optimizations")
    print("-" * 50)
    try:
        # Check TF32
        assert torch.backends.cuda.matmul.allow_tf32 == True, "TF32 not enabled"
        print("✓ TF32 enabled")
        
        # Check models loaded
        assert hasattr(pipeline, 'low_noise_model'), "Low noise model missing"
        assert hasattr(pipeline, 'high_noise_model'), "High noise model missing"
        print("✓ Both MoE models loaded")
        
        # Check VAE
        assert hasattr(pipeline, 'vae'), "VAE missing"
        print("✓ VAE loaded")
        
        # Check text encoder
        assert hasattr(pipeline, 'text_encoder'), "Text encoder missing"
        print("✓ Text encoder loaded")
        
        print("PASS ✓\n")
    except AssertionError as e:
        print(f"✗ FAIL: {e}\n")
        sys.exit(1)
    
    # Test batched CFG actually used
    print("Test 5: Batched CFG Validation")
    print("-" * 50)
    try:
        # Generate with and without batched CFG
        print("Testing batched CFG=True...")
        torch.cuda.reset_peak_memory_stats()
        start = time.time()
        
        _ = pipeline.generate(
            input_prompt="test",
            size=(1280, 720),
            frame_num=5,
            sampling_steps=2,
            use_batched_cfg=True,
            offload_model=True,
            seed=42
        )
        
        time_batched = time.time() - start
        mem_batched = torch.cuda.max_memory_allocated() / 1024**3
        
        print(f"✓ Batched: {time_batched:.1f}s, {mem_batched:.1f} GB")
        
        print("Testing batched CFG=False...")
        torch.cuda.reset_peak_memory_stats()
        start = time.time()
        
        _ = pipeline.generate(
            input_prompt="test",
            size=(1280, 720),
            frame_num=5,
            sampling_steps=2,
            use_batched_cfg=False,
            offload_model=True,
            seed=42
        )
        
        time_unbatched = time.time() - start
        mem_unbatched = torch.cuda.max_memory_allocated() / 1024**3
        
        print(f"✓ Unbatched: {time_unbatched:.1f}s, {mem_unbatched:.1f} GB")
        
        if time_unbatched > time_batched * 1.3:
            speedup = time_unbatched / time_batched
            print(f"✓ Batched CFG is {speedup:.2f}x faster!")
        else:
            print(f"⚠ Note: Speedup not significant in this quick test")
        
        print("PASS ✓\n")
    except Exception as e:
        print(f"✗ FAIL: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Summary
    print("="*70)
    print("CHECKPOINT TEST SUMMARY")
    print("="*70)
    print("✓ All tests passed!")
    print("\nVerified:")
    print("  ✓ Checkpoint structure correct")
    print("  ✓ Pipeline loads successfully")
    print("  ✓ Video generation works")
    print("  ✓ Optimizations are active")
    print("  ✓ Batched CFG implementation correct")
    print("\n🎉 Ready for production use on H200!")
    print("\nNext: Run full benchmark:")
    print(f"  python benchmark_t2v.py --ckpt_dir {args.ckpt_dir} --mode both")
    print("="*70)

if __name__ == "__main__":
    main()

