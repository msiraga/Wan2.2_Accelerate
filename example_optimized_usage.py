#!/usr/bin/env python3
"""
Simple example showing how to use the optimized T2V pipeline.

This demonstrates the three main usage patterns:
1. Direct import and use
2. Comparing performance
3. Customizing optimization levels
"""

import torch
from wan.configs import WAN_CONFIGS, SIZE_CONFIGS


def example_basic_usage():
    """Basic usage: Generate a video with default optimizations."""
    print("=" * 80)
    print("EXAMPLE 1: Basic Usage")
    print("=" * 80)
    
    from wan.text2video_optimized import WanT2VOptimized
    
    # Configuration
    config = WAN_CONFIGS["t2v-A14B"]
    checkpoint_dir = "/path/to/your/checkpoints"  # UPDATE THIS
    
    # Create pipeline with optimizations enabled
    pipeline = WanT2VOptimized(
        config=config,
        checkpoint_dir=checkpoint_dir,
        device_id=0,
        rank=0,
        enable_compile=True,      # torch.compile for 1.2x speedup
        enable_tf32=True,         # TF32 for 1.3x speedup (Ampere+)
        compile_mode="reduce-overhead",  # Balanced compilation
    )
    
    # Generate video
    video = pipeline.generate(
        input_prompt="Two anthropomorphic cats in boxing gear fighting on stage",
        size=(1280, 720),
        frame_num=81,
        sampling_steps=40,
        guide_scale=(3.0, 4.0),
        seed=42,
        use_batched_cfg=True,     # Batched CFG for 1.8x speedup
        offload_model=False,      # Keep models on GPU (requires 80GB)
    )
    
    print(f"Generated video shape: {video.shape}")
    print("✓ Video generation complete!\n")
    
    return video


def example_memory_constrained():
    """Memory-constrained usage: For GPUs with less VRAM."""
    print("=" * 80)
    print("EXAMPLE 2: Memory-Constrained Mode (24GB+ VRAM)")
    print("=" * 80)
    
    from wan.text2video_optimized import WanT2VOptimized
    
    config = WAN_CONFIGS["t2v-A14B"]
    checkpoint_dir = "/path/to/your/checkpoints"  # UPDATE THIS
    
    # Create pipeline with memory-saving settings
    pipeline = WanT2VOptimized(
        config=config,
        checkpoint_dir=checkpoint_dir,
        device_id=0,
        rank=0,
        enable_compile=False,     # Disable to save memory
        enable_tf32=True,         # Still use TF32 (no memory cost)
        t5_cpu=True,             # Keep T5 on CPU
    )
    
    # Generate with model offloading
    video = pipeline.generate(
        input_prompt="A serene lake at sunset",
        size=(960, 544),          # Smaller resolution
        frame_num=49,             # Fewer frames
        sampling_steps=40,
        use_batched_cfg=True,     # Still use batched CFG
        offload_model=True,       # Offload models to CPU
    )
    
    print(f"Generated video shape: {video.shape}")
    print(f"Peak VRAM: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
    print("✓ Memory-efficient generation complete!\n")
    
    return video


def example_maximum_performance():
    """Maximum performance: For H100/A100 80GB."""
    print("=" * 80)
    print("EXAMPLE 3: Maximum Performance Mode (80GB VRAM)")
    print("=" * 80)
    
    from wan.text2video_optimized import WanT2VOptimized
    import time
    
    config = WAN_CONFIGS["t2v-A14B"]
    checkpoint_dir = "/path/to/your/checkpoints"  # UPDATE THIS
    
    # Create pipeline with aggressive optimizations
    pipeline = WanT2VOptimized(
        config=config,
        checkpoint_dir=checkpoint_dir,
        device_id=0,
        rank=0,
        enable_compile=True,
        enable_tf32=True,
        compile_mode="max-autotune",  # Maximum optimization (slow first run)
    )
    
    print("Note: First run will be slower due to compilation overhead\n")
    
    # Warmup (compilation happens here)
    print("Warmup run (compiling models)...")
    torch.cuda.synchronize()
    warmup_start = time.time()
    
    _ = pipeline.generate(
        input_prompt="A cat",
        size=(1280, 720),
        frame_num=81,
        sampling_steps=40,
        use_batched_cfg=True,
        offload_model=False,
    )
    
    torch.cuda.synchronize()
    warmup_time = time.time() - warmup_start
    print(f"Warmup time: {warmup_time:.2f}s (includes compilation)\n")
    
    # Actual generation (using compiled models)
    print("Actual generation (using compiled models)...")
    torch.cuda.synchronize()
    start = time.time()
    
    video = pipeline.generate(
        input_prompt="Two anthropomorphic cats in boxing gear fighting intensely on a spotlit stage",
        size=(1280, 720),
        frame_num=81,
        sampling_steps=40,
        guide_scale=(3.0, 4.0),
        seed=42,
        use_batched_cfg=True,
        offload_model=False,
    )
    
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    print(f"Generated video shape: {video.shape}")
    print(f"Generation time: {elapsed:.2f}s")
    print(f"Peak VRAM: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
    print("✓ Maximum performance generation complete!\n")
    
    return video


def example_compare_original_vs_optimized():
    """Compare original vs optimized implementation."""
    print("=" * 80)
    print("EXAMPLE 4: Comparing Original vs Optimized")
    print("=" * 80)
    
    import time
    
    config = WAN_CONFIGS["t2v-A14B"]
    checkpoint_dir = "/path/to/your/checkpoints"  # UPDATE THIS
    prompt = "A beautiful sunset over mountains"
    
    # Test original
    print("Testing original implementation...")
    from wan.text2video import WanT2V
    
    pipeline_orig = WanT2V(
        config=config,
        checkpoint_dir=checkpoint_dir,
        device_id=0,
        rank=0,
    )
    
    torch.cuda.synchronize()
    start = time.time()
    
    video_orig = pipeline_orig.generate(
        input_prompt=prompt,
        size=(1280, 720),
        frame_num=81,
        sampling_steps=40,
        seed=42,
        offload_model=False,
    )
    
    torch.cuda.synchronize()
    time_orig = time.time() - start
    mem_orig = torch.cuda.max_memory_allocated() / 1024**3
    
    print(f"Original time: {time_orig:.2f}s")
    print(f"Original VRAM: {mem_orig:.2f} GB\n")
    
    # Cleanup
    del pipeline_orig
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # Test optimized
    print("Testing optimized implementation...")
    from wan.text2video_optimized import WanT2VOptimized
    
    pipeline_opt = WanT2VOptimized(
        config=config,
        checkpoint_dir=checkpoint_dir,
        device_id=0,
        rank=0,
        enable_compile=True,
        enable_tf32=True,
    )
    
    # Warmup
    _ = pipeline_opt.generate(
        input_prompt=prompt,
        size=(1280, 720),
        frame_num=81,
        sampling_steps=40,
        seed=42,
        use_batched_cfg=True,
        offload_model=False,
    )
    
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.time()
    
    video_opt = pipeline_opt.generate(
        input_prompt=prompt,
        size=(1280, 720),
        frame_num=81,
        sampling_steps=40,
        seed=42,
        use_batched_cfg=True,
        offload_model=False,
    )
    
    torch.cuda.synchronize()
    time_opt = time.time() - start
    mem_opt = torch.cuda.max_memory_allocated() / 1024**3
    
    print(f"Optimized time: {time_opt:.2f}s")
    print(f"Optimized VRAM: {mem_opt:.2f} GB\n")
    
    # Compare
    speedup = time_orig / time_opt
    mem_delta = mem_opt - mem_orig
    
    print("COMPARISON:")
    print(f"  Speedup: {speedup:.2f}x")
    print(f"  Time saved: {time_orig - time_opt:.2f}s")
    print(f"  VRAM delta: {mem_delta:+.2f} GB")
    print("✓ Comparison complete!\n")


def example_custom_optimization_levels():
    """Customize which optimizations to use."""
    print("=" * 80)
    print("EXAMPLE 5: Custom Optimization Levels")
    print("=" * 80)
    
    from wan.text2video_optimized import WanT2VOptimized
    
    config = WAN_CONFIGS["t2v-A14B"]
    checkpoint_dir = "/path/to/your/checkpoints"  # UPDATE THIS
    
    # Level 1: Conservative (only batched CFG)
    print("Level 1: Conservative (only batched CFG)")
    pipeline_l1 = WanT2VOptimized(
        config=config,
        checkpoint_dir=checkpoint_dir,
        enable_compile=False,
        enable_tf32=False,
    )
    
    video_l1 = pipeline_l1.generate(
        input_prompt="A cat",
        use_batched_cfg=True,
        offload_model=True,
    )
    print("✓ Expected: ~1.8x speedup\n")
    
    # Level 2: Moderate (batched CFG + TF32)
    print("Level 2: Moderate (batched CFG + TF32)")
    pipeline_l2 = WanT2VOptimized(
        config=config,
        checkpoint_dir=checkpoint_dir,
        enable_compile=False,
        enable_tf32=True,
    )
    
    video_l2 = pipeline_l2.generate(
        input_prompt="A cat",
        use_batched_cfg=True,
        offload_model=True,
    )
    print("✓ Expected: ~2.3x speedup\n")
    
    # Level 3: Aggressive (all optimizations)
    print("Level 3: Aggressive (all optimizations)")
    pipeline_l3 = WanT2VOptimized(
        config=config,
        checkpoint_dir=checkpoint_dir,
        enable_compile=True,
        enable_tf32=True,
        compile_mode="max-autotune",
    )
    
    video_l3 = pipeline_l3.generate(
        input_prompt="A cat",
        use_batched_cfg=True,
        offload_model=False,
    )
    print("✓ Expected: ~3.3x speedup\n")


def main():
    """Run all examples (comment out as needed)."""
    
    print("\n" + "=" * 80)
    print("WAN 2.2 T2V OPTIMIZATION EXAMPLES")
    print("=" * 80 + "\n")
    
    print("⚠️  WARNING: These examples require:")
    print("  1. Valid checkpoint directory path")
    print("  2. Appropriate GPU (see hardware recommendations)")
    print("  3. Sufficient VRAM for chosen optimization level\n")
    
    print("📝 NOTE: Comment out examples you don't want to run\n")
    
    # Uncomment the examples you want to run:
    
    # example_basic_usage()
    # example_memory_constrained()
    # example_maximum_performance()
    # example_compare_original_vs_optimized()
    # example_custom_optimization_levels()
    
    print("=" * 80)
    print("To run examples, uncomment them in main() and update checkpoint_dir")
    print("=" * 80)


if __name__ == "__main__":
    main()

