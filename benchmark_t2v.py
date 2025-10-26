#!/usr/bin/env python3
"""
Benchmark script to compare original vs optimized T2V implementation.

Usage:
    python benchmark_t2v.py --ckpt_dir /path/to/checkpoints --mode both
    python benchmark_t2v.py --ckpt_dir /path/to/checkpoints --mode optimized --compile --no-offload
"""

import argparse
import time
import torch
import logging
import sys
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(stream=sys.stdout)]
)

def benchmark_generation(pipeline, prompt, size, frame_num, **kwargs):
    """Run a single generation and measure time."""
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    
    start_time = time.time()
    
    video = pipeline.generate(
        prompt,
        size=size,
        frame_num=frame_num,
        **kwargs
    )
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    elapsed = end_time - start_time
    peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
    
    return elapsed, peak_memory, video


def run_benchmark(args):
    """Run benchmark comparing original vs optimized."""
    
    from wan.configs import WAN_CONFIGS, SIZE_CONFIGS
    
    # Configuration
    config = WAN_CONFIGS["t2v-A14B"]
    size = SIZE_CONFIGS[args.size]
    prompt = args.prompt or "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
    
    results = {}
    
    # Test configurations
    if args.mode in ["original", "both"]:
        logging.info("=" * 80)
        logging.info("BENCHMARKING ORIGINAL IMPLEMENTATION")
        logging.info("=" * 80)
        
        from wan.text2video import WanT2V
        
        logging.info("Loading models...")
        pipeline = WanT2V(
            config=config,
            checkpoint_dir=args.ckpt_dir,
            device_id=0,
            rank=0,
            t5_cpu=args.t5_cpu,
            convert_model_dtype=args.convert_dtype,
        )
        
        # Warmup
        if args.warmup:
            logging.info("Warmup run...")
            _ = benchmark_generation(
                pipeline, prompt, size, args.frame_num,
                shift=args.shift,
                sample_solver=args.solver,
                sampling_steps=args.steps,
                guide_scale=args.guide_scale,
                seed=args.seed,
                offload_model=args.offload_model
            )
        
        # Benchmark runs
        times = []
        memories = []
        for i in range(args.num_runs):
            logging.info(f"Run {i+1}/{args.num_runs}...")
            elapsed, peak_mem, video = benchmark_generation(
                pipeline, prompt, size, args.frame_num,
                shift=args.shift,
                sample_solver=args.solver,
                sampling_steps=args.steps,
                guide_scale=args.guide_scale,
                seed=args.seed,
                offload_model=args.offload_model
            )
            times.append(elapsed)
            memories.append(peak_mem)
            logging.info(f"  Time: {elapsed:.2f}s, Peak Memory: {peak_mem:.2f} GB")
        
        avg_time = sum(times) / len(times)
        avg_memory = sum(memories) / len(memories)
        
        results["original"] = {
            "times": times,
            "avg_time": avg_time,
            "memories": memories,
            "avg_memory": avg_memory
        }
        
        logging.info(f"\nOriginal Average: {avg_time:.2f}s, {avg_memory:.2f} GB")
        
        # Cleanup
        del pipeline
        torch.cuda.empty_cache()
    
    if args.mode in ["optimized", "both"]:
        logging.info("\n" + "=" * 80)
        logging.info("BENCHMARKING OPTIMIZED IMPLEMENTATION")
        logging.info("=" * 80)
        
        from wan.text2video_optimized import WanT2VOptimized
        
        logging.info("Loading models...")
        pipeline = WanT2VOptimized(
            config=config,
            checkpoint_dir=args.ckpt_dir,
            device_id=0,
            rank=0,
            t5_cpu=args.t5_cpu,
            convert_model_dtype=args.convert_dtype,
            enable_compile=args.compile,
            enable_tf32=args.tf32,
            compile_mode=args.compile_mode,
        )
        
        # Warmup (especially important for torch.compile)
        if args.warmup or args.compile:
            logging.info("Warmup run (includes compilation if enabled)...")
            _ = benchmark_generation(
                pipeline, prompt, size, args.frame_num,
                shift=args.shift,
                sample_solver=args.solver,
                sampling_steps=args.steps,
                guide_scale=args.guide_scale,
                seed=args.seed,
                offload_model=args.offload_model
            )
        
        # Benchmark runs
        times = []
        memories = []
        for i in range(args.num_runs):
            logging.info(f"Run {i+1}/{args.num_runs}...")
            elapsed, peak_mem, video = benchmark_generation(
                pipeline, prompt, size, args.frame_num,
                shift=args.shift,
                sample_solver=args.solver,
                sampling_steps=args.steps,
                guide_scale=args.guide_scale,
                seed=args.seed,
                offload_model=args.offload_model
            )
            times.append(elapsed)
            memories.append(peak_mem)
            logging.info(f"  Time: {elapsed:.2f}s, Peak Memory: {peak_mem:.2f} GB")
        
        avg_time = sum(times) / len(times)
        avg_memory = sum(memories) / len(memories)
        
        results["optimized"] = {
            "times": times,
            "avg_time": avg_time,
            "memories": memories,
            "avg_memory": avg_memory
        }
        
        logging.info(f"\nOptimized Average: {avg_time:.2f}s, {avg_memory:.2f} GB")
        
        # Cleanup
        del pipeline
        torch.cuda.empty_cache()
    
    # Print comparison
    if args.mode == "both":
        logging.info("\n" + "=" * 80)
        logging.info("COMPARISON RESULTS")
        logging.info("=" * 80)
        
        orig_time = results["original"]["avg_time"]
        opt_time = results["optimized"]["avg_time"]
        speedup = orig_time / opt_time
        
        orig_mem = results["original"]["avg_memory"]
        opt_mem = results["optimized"]["avg_memory"]
        mem_delta = opt_mem - orig_mem
        
        logging.info(f"Original Implementation:")
        logging.info(f"  Average Time: {orig_time:.2f}s")
        logging.info(f"  Peak Memory:  {orig_mem:.2f} GB")
        
        logging.info(f"\nOptimized Implementation:")
        logging.info(f"  Average Time: {opt_time:.2f}s")
        logging.info(f"  Peak Memory:  {opt_mem:.2f} GB")
        
        logging.info(f"\nPerformance Gain:")
        logging.info(f"  Speedup:      {speedup:.2f}x")
        logging.info(f"  Time Saved:   {orig_time - opt_time:.2f}s ({(1-1/speedup)*100:.1f}%)")
        logging.info(f"  Memory Delta: {mem_delta:+.2f} GB")
        
        # Optimization breakdown
        logging.info(f"\nOptimizations Applied:")
        if args.tf32:
            logging.info(f"  ✓ TF32 matmul")
        if args.compile:
            logging.info(f"  ✓ torch.compile (mode={args.compile_mode})")
        if not args.offload_model:
            logging.info(f"  ✓ Models kept on GPU")
    
    # Save results
    if args.save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"benchmark_results_{timestamp}.txt"
        
        with open(results_file, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("WAN 2.2 T2V BENCHMARK RESULTS\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Configuration:\n")
            f.write(f"  Size: {args.size}\n")
            f.write(f"  Frames: {args.frame_num}\n")
            f.write(f"  Steps: {args.steps}\n")
            f.write(f"  Solver: {args.solver}\n")
            f.write(f"  Offload: {args.offload_model}\n\n")
            
            if "original" in results:
                f.write(f"Original Implementation:\n")
                f.write(f"  Average Time: {results['original']['avg_time']:.2f}s\n")
                f.write(f"  Peak Memory:  {results['original']['avg_memory']:.2f} GB\n\n")
            
            if "optimized" in results:
                f.write(f"Optimized Implementation:\n")
                f.write(f"  Average Time: {results['optimized']['avg_time']:.2f}s\n")
                f.write(f"  Peak Memory:  {results['optimized']['avg_memory']:.2f} GB\n\n")
            
            if args.mode == "both":
                speedup = results["original"]["avg_time"] / results["optimized"]["avg_time"]
                f.write(f"Speedup: {speedup:.2f}x\n")
        
        logging.info(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark T2V performance")
    
    # Required
    parser.add_argument("--ckpt_dir", type=str, required=True,
                       help="Path to checkpoint directory")
    
    # Benchmark mode
    parser.add_argument("--mode", type=str, default="both",
                       choices=["original", "optimized", "both"],
                       help="Which implementation to benchmark")
    
    # Generation params
    parser.add_argument("--prompt", type=str, default=None,
                       help="Text prompt for generation")
    parser.add_argument("--size", type=str, default="1280*720",
                       help="Video resolution")
    parser.add_argument("--frame_num", type=int, default=81,
                       help="Number of frames")
    parser.add_argument("--steps", type=int, default=40,
                       help="Sampling steps")
    parser.add_argument("--shift", type=float, default=12.0,
                       help="Sampling shift")
    parser.add_argument("--guide_scale", type=float, default=5.0,
                       help="CFG scale (default: 5.0)")
    parser.add_argument("--solver", type=str, default="unipc",
                       choices=["unipc", "dpm++"],
                       help="Sampling solver")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    # Model config
    parser.add_argument("--offload_model", action="store_true", default=False,
                       help="Offload models to CPU (slower but uses less VRAM)")
    parser.add_argument("--t5_cpu", action="store_true", default=False,
                       help="Keep T5 on CPU")
    parser.add_argument("--convert_dtype", action="store_true", default=False,
                       help="Convert model dtype")
    
    # Optimization flags
    parser.add_argument("--compile", action="store_true", default=True,
                       help="Enable torch.compile")
    parser.add_argument("--no-compile", dest="compile", action="store_false",
                       help="Disable torch.compile")
    parser.add_argument("--compile_mode", type=str, default="reduce-overhead",
                       choices=["default", "reduce-overhead", "max-autotune"],
                       help="torch.compile mode")
    parser.add_argument("--tf32", action="store_true", default=True,
                       help="Enable TF32")
    parser.add_argument("--no-tf32", dest="tf32", action="store_false",
                       help="Disable TF32")
    
    # Benchmark config
    parser.add_argument("--num_runs", type=int, default=3,
                       help="Number of benchmark runs")
    parser.add_argument("--warmup", action="store_true", default=True,
                       help="Run warmup before benchmark")
    parser.add_argument("--no-warmup", dest="warmup", action="store_false",
                       help="Skip warmup")
    parser.add_argument("--save_results", action="store_true", default=True,
                       help="Save results to file")
    
    args = parser.parse_args()
    
    run_benchmark(args)

