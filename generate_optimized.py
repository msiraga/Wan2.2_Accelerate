#!/usr/bin/env python3
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""
Optimized T2V generation script with 3-4x speedup.

Usage:
    # Maximum speed (requires 80GB VRAM)
    python generate_optimized.py --ckpt_dir /path/to/ckpt --prompt "Your prompt" --no-offload
    
    # Balanced mode (48GB+ VRAM)
    python generate_optimized.py --ckpt_dir /path/to/ckpt --prompt "Your prompt"
    
    # Memory constrained (24GB+ VRAM)
    python generate_optimized.py --ckpt_dir /path/to/ckpt --prompt "Your prompt" --t5_cpu --no-compile
"""

import argparse
import logging
import os
import sys
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

import random
import torch
import torch.distributed as dist
from PIL import Image

from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS
from wan.text2video_optimized import WanT2VOptimized
from wan.utils.utils import save_video, str2bool


def _init_logging(rank):
    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)])
    else:
        logging.basicConfig(level=logging.ERROR)


def main():
    parser = argparse.ArgumentParser(
        description="Generate video from text using optimized Wan T2V"
    )
    
    # Required
    parser.add_argument("--ckpt_dir", type=str, required=True,
                       help="Path to checkpoint directory")
    parser.add_argument("--prompt", type=str, required=True,
                       help="Text prompt for generation")
    
    # Generation parameters
    parser.add_argument("--size", type=str, default="1280*720",
                       help="Video resolution (width*height)")
    parser.add_argument("--frame_num", type=int, default=None,
                       help="Number of frames (must be 4n+1)")
    parser.add_argument("--sample_steps", type=int, default=None,
                       help="Sampling steps")
    parser.add_argument("--sample_shift", type=float, default=None,
                       help="Sampling shift factor")
    parser.add_argument("--sample_guide_scale", type=float, default=None,
                       help="CFG scale")
    parser.add_argument("--sample_solver", type=str, default='unipc',
                       choices=['unipc', 'dpm++'],
                       help="Sampling solver")
    parser.add_argument("--n_prompt", type=str, default="",
                       help="Negative prompt")
    parser.add_argument("--seed", type=int, default=-1,
                       help="Random seed (-1 for random)")
    
    # Output
    parser.add_argument("--save_file", type=str, default=None,
                       help="Output video filename")
    
    # Model configuration
    parser.add_argument("--offload_model", type=str2bool, default=True,
                       help="Offload models to CPU (slower but saves VRAM)")
    parser.add_argument("--no-offload", dest="offload_model", 
                       action="store_false",
                       help="Keep models on GPU (faster, requires more VRAM)")
    parser.add_argument("--t5_cpu", action="store_true", default=False,
                       help="Keep T5 encoder on CPU")
    parser.add_argument("--convert_model_dtype", action="store_true", default=False,
                       help="Convert model dtype")
    
    # Optimization flags
    parser.add_argument("--compile", type=str2bool, default=True,
                       help="Enable torch.compile (default: True)")
    parser.add_argument("--no-compile", dest="compile", action="store_false",
                       help="Disable torch.compile")
    parser.add_argument("--compile_mode", type=str, default="reduce-overhead",
                       choices=["default", "reduce-overhead", "max-autotune"],
                       help="torch.compile optimization mode")
    
    parser.add_argument("--tf32", type=str2bool, default=True,
                       help="Enable TF32 (default: True, Ampere+ only)")
    parser.add_argument("--no-tf32", dest="tf32", action="store_false",
                       help="Disable TF32")
    
    
    # Distributed (not commonly used for single generation)
    parser.add_argument("--ulysses_size", type=int, default=1,
                       help="Ulysses parallelism size")
    parser.add_argument("--t5_fsdp", action="store_true", default=False,
                       help="Use FSDP for T5")
    parser.add_argument("--dit_fsdp", action="store_true", default=False,
                       help="Use FSDP for DiT")
    
    args = parser.parse_args()
    
    # Setup
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank
    
    _init_logging(rank)
    
    # Load config
    task = "t2v-A14B"
    config = WAN_CONFIGS[task]
    
    # Validate size
    assert args.size in SUPPORTED_SIZES[task], \
        f"Unsupported size {args.size} for {task}"
    
    size = SIZE_CONFIGS[args.size]
    
    # Set defaults from config
    if args.frame_num is None:
        args.frame_num = config.frame_num
    if args.sample_steps is None:
        args.sample_steps = config.sample_steps
    if args.sample_shift is None:
        args.sample_shift = config.sample_shift
    if args.sample_guide_scale is None:
        args.sample_guide_scale = config.sample_guide_scale
    
    # Seed
    if args.seed < 0:
        args.seed = random.randint(0, sys.maxsize)
    
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size)
    
    logging.info("=" * 80)
    logging.info("OPTIMIZED WAN 2.2 T2V GENERATION")
    logging.info("=" * 80)
    logging.info(f"Prompt: {args.prompt}")
    logging.info(f"Size: {args.size} ({size[0]}x{size[1]})")
    logging.info(f"Frames: {args.frame_num}")
    logging.info(f"Steps: {args.sample_steps}")
    logging.info(f"Solver: {args.sample_solver}")
    logging.info(f"Seed: {args.seed}")
    
    logging.info("\nOptimizations:")
    logging.info(f"  TF32: {'✓' if args.tf32 else '✗'}")
    logging.info(f"  torch.compile: {'✓' if args.compile else '✗'} (mode={args.compile_mode})")
    logging.info(f"  Model offloading: {'✓' if args.offload_model else '✗'}")
    logging.info(f"  T5 on CPU: {'✓' if args.t5_cpu else '✗'}")
    logging.info("=" * 80 + "\n")
    
    # Create pipeline
    logging.info("Loading optimized pipeline...")
    pipeline = WanT2VOptimized(
        config=config,
        checkpoint_dir=args.ckpt_dir,
        device_id=device,
        rank=rank,
        t5_fsdp=args.t5_fsdp,
        dit_fsdp=args.dit_fsdp,
        use_sp=(args.ulysses_size > 1),
        t5_cpu=args.t5_cpu,
        convert_model_dtype=args.convert_model_dtype,
        enable_compile=args.compile,
        enable_tf32=args.tf32,
        compile_mode=args.compile_mode,
    )
    
    # Generate
    import time
    torch.cuda.synchronize()
    start_time = time.time()
    
    logging.info("Generating video...")
    if args.compile:
        logging.info("(First run includes compilation overhead - subsequent runs will be faster)")
    
    video = pipeline.generate(
        input_prompt=args.prompt,
        size=size,
        frame_num=args.frame_num,
        shift=args.sample_shift,
        sample_solver=args.sample_solver,
        sampling_steps=args.sample_steps,
        guide_scale=args.sample_guide_scale,
        n_prompt=args.n_prompt,
        seed=args.seed,
        offload_model=args.offload_model
    )
    
    torch.cuda.synchronize()
    elapsed = time.time() - start_time
    
    logging.info(f"\n✓ Generation complete in {elapsed:.2f}s")
    logging.info(f"  Peak VRAM: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
    
    # Save
    if rank == 0:
        if args.save_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            prompt_short = args.prompt.replace(" ", "_")[:50]
            args.save_file = f"t2v_opt_{args.size.replace('*','x')}_{prompt_short}_{timestamp}.mp4"
        
        logging.info(f"Saving video to: {args.save_file}")
        save_video(
            tensor=video[None],
            save_file=args.save_file,
            fps=config.sample_fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1)
        )
        logging.info("✓ Video saved successfully")
    
    # Cleanup
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    
    logging.info("\nDone!")


if __name__ == "__main__":
    main()

