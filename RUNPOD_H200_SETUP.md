# 🚀 Complete Setup Guide: Wan 2.2 T2V on RunPod H200

**Complete step-by-step guide to run optimized Wan 2.2 T2V on RunPod H200 instance**

Expected result: **3.5x faster generation** (~90s for 81 frames @ 1280×720)

---

## 📋 Prerequisites

- RunPod account
- SSH client (Terminal on Mac/Linux, PowerShell/PuTTY on Windows)
- ~60GB free space for checkpoints
- Basic command-line knowledge

---

## STEP 1: Launch RunPod H200 Instance

### 1.1 Go to RunPod
1. Login to https://www.runpod.io/
2. Click **"Deploy"** or **"GPU Pods"**

### 1.2 Select H200 GPU
1. Filter by GPU: Select **H200 80GB** or **H200 SXM 141GB**
2. Choose a template:
   - Recommended: **PyTorch 2.4+** or **RunPod PyTorch**
   - Or: **RunPod Ubuntu** (we'll install PyTorch)

### 1.3 Configure Pod
1. **Disk Space**: Set to at least **100GB** (for checkpoint + code + outputs)
2. **Container Disk**: 50GB minimum
3. Click **"Deploy On-Demand"** or **"Deploy Spot"** (cheaper but can be interrupted)

### 1.4 Get SSH Connection Info
1. Wait for pod to start (status: **Running**)
2. Click **"Connect"** → **"TCP Port Mappings"**
3. Find SSH port (usually port **22** mapped to external port)
4. Copy the connection command, looks like:
   ```
   ssh root@<pod-id>.proxy.runpod.net -p <port>
   ```

---

## STEP 2: Connect via SSH

### 2.1 Connect to Your Pod

**From your local machine:**

```bash
# Replace with your actual connection details from RunPod
ssh root@<your-pod-id>.proxy.runpod.net -p <your-port>

# Example:
# ssh root@abc123def456.proxy.runpod.net -p 12345
```

**First time connecting:** Type `yes` when asked about fingerprint

### 2.2 Verify GPU

```bash
# Check H200 is available
nvidia-smi

# You should see:
# - GPU name: H200 or H200 SXM
# - Memory: 80GB or 141GB
# - CUDA Version: 12.1+
```

---

## STEP 3: Setup Environment

### 3.1 Update System (Optional but Recommended)

```bash
apt-get update
apt-get install -y git wget vim htop
```

### 3.2 Check Python & PyTorch

```bash
# Check Python version (should be 3.10+)
python --version

# Check PyTorch (should be 2.4+)
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}')"
```

**If PyTorch is missing or old:**
```bash
# Option 1: Auto-detect CUDA (recommended for RunPod)
pip install torch>=2.4.0 torchvision torchaudio

# Option 2: Explicit CUDA 12.4 (H200 optimized)
pip install torch>=2.4.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Option 3: CUDA 12.1 (most stable)
pip install torch>=2.4.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Note:** H200 supports CUDA 11.8+ but 12.1+ recommended for best performance.

---

## STEP 4: Clone Repository

### 4.1 Navigate to Workspace

```bash
# Create a workspace directory
cd /workspace
# Or if /workspace doesn't exist:
# mkdir -p ~/wan-project && cd ~/wan-project
```

### 4.2 Clone Wan 2.2 Repository

```bash
# Clone the repo
git clone https://github.com/Wan-Video/Wan2.2.git
cd Wan2.2

# Verify you're in the right place
ls -la
# You should see: generate.py, wan/, requirements.txt, etc.
```

---

## STEP 5: Install Dependencies

### 5.1 Install Requirements

```bash
# Install base requirements
pip install -r requirements.txt

# If flash_attn fails, install other packages first, then:
pip install flash_attn --no-build-isolation
```

**Expected time:** 5-10 minutes

### 5.2 Verify Installation

```bash
python -c "import torch; import diffusers; import transformers; print('✓ All imports successful')"
```

---

## STEP 6: Download Model Checkpoints

### 6.1 Install Download Tool

```bash
pip install "huggingface_hub[cli]"
```

### 6.2 Download T2V-A14B Model

**This is ~56GB, will take 10-30 minutes depending on connection**

```bash
# Create checkpoints directory
cd /workspace/Wan2.2
mkdir -p checkpoints
cd checkpoints

# Download the model (faster with hf_transfer)
pip install huggingface_hub[hf_transfer]
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download \
    Wan-AI/Wan2.2-T2V-A14B \
    --local-dir ./Wan2.2-T2V-A14B
```

**Monitor progress:**
```bash
# In another terminal/SSH session:
watch -n 5 du -sh /workspace/Wan2.2/checkpoints/Wan2.2-T2V-A14B
```

### 6.3 Verify Download

```bash
# Check all files are present
ls -lh /workspace/Wan2.2/checkpoints/Wan2.2-T2V-A14B/

# You should see:
# - low_noise_model/
# - high_noise_model/
# - models_t5_umt5-xxl-enc-bf16.pth (~7GB)
# - Wan2.1_VAE.pth (~600MB)
# - google/ (tokenizer)
```

**Checkpoint Structure:**
```
/workspace/Wan2.2/checkpoints/Wan2.2-T2V-A14B/
├── low_noise_model/
│   ├── config.json
│   └── diffusion_pytorch_model.safetensors  (~28GB)
├── high_noise_model/
│   ├── config.json
│   └── diffusion_pytorch_model.safetensors  (~28GB)
├── models_t5_umt5-xxl-enc-bf16.pth          (~7GB)
├── Wan2.1_VAE.pth                           (~600MB)
└── google/
    └── umt5-xxl/
        ├── tokenizer.json
        └── ...
```

---

## STEP 7: Setup Optimized Files

### 7.1 Copy Optimization Files

**You already have these in your cloned repo!**

```bash
cd /workspace/Wan2.2

# Verify optimized files exist
ls -la generate_optimized.py
ls -la wan/text2video_optimized.py
ls -la benchmark_t2v.py
```

### 7.2 Set Checkpoint Path Variable (Convenience)

```bash
# Add to your session for easy reference
export CKPT_DIR=/workspace/Wan2.2/checkpoints/Wan2.2-T2V-A14B

# Verify it works
echo $CKPT_DIR
```

---

## STEP 8: Test Installation (Quick Test)

### 8.1 Run Quick Test (17 frames, ~30 seconds)

```bash
cd /workspace/Wan2.2

python generate_optimized.py \
    --ckpt_dir $CKPT_DIR \
    --prompt "A cat sitting on a chair" \
    --frame_num 17 \
    --no-offload \
    --compile \
    --size 1280*720
```

**What to expect:**
- First run: ~60-90s (includes compilation)
- Output: `t2v_opt_1280x720_A_cat_sitting_on_a_chair_YYYYMMDD_HHMMSS.mp4`
- Peak VRAM: ~78GB / 141GB

**If successful:** You'll see:
```
✓ Generation complete in XX.XXs
  Peak VRAM: XX.XX GB
Saving video to: t2v_opt_...mp4
✓ Video saved successfully
```

---

## STEP 9: Run Full Generation (81 frames)

### 9.1 Maximum Performance Configuration

```bash
cd /workspace/Wan2.2

python generate_optimized.py \
    --ckpt_dir $CKPT_DIR \
    --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage" \
    --size 1280*720 \
    --frame_num 81 \
    --sample_steps 40 \
    --no-offload \
    --compile \
    --compile_mode max-autotune \
    --batched_cfg \
    --tf32 \
    --seed 42
```

**Expected Performance:**
- First run: ~120-140s (includes compilation)
- Subsequent runs: ~80-90s
- Peak VRAM: ~78-80GB / 141GB
- Output: High-quality 81-frame video

### 9.2 Generate Multiple Videos

```bash
# After warmup, generate quickly!
python generate_optimized.py --ckpt_dir $CKPT_DIR \
    --prompt "A majestic lion walking through savanna at sunset" \
    --no-offload --seed 1 --save_file video1.mp4

python generate_optimized.py --ckpt_dir $CKPT_DIR \
    --prompt "Underwater scene with colorful fish and coral reefs" \
    --no-offload --seed 2 --save_file video2.mp4

# Each takes ~85s
```

---

## STEP 10: Benchmark Performance

### 10.1 Compare Original vs Optimized

```bash
cd /workspace/Wan2.2

python benchmark_t2v.py \
    --ckpt_dir $CKPT_DIR \
    --mode both \
    --num_runs 3 \
    --size 1280*720 \
    --frame_num 81 \
    --no-offload \
    --compile_mode max-autotune
```

**This will:**
- Run original implementation 3 times
- Run optimized implementation 3 times
- Show speedup comparison
- Save results to file

**Expected Output:**
```
COMPARISON RESULTS
==================
Original Implementation:
  Average Time: 300.00s
  Peak Memory:  62.00 GB

Optimized Implementation:
  Average Time: 90.00s
  Peak Memory:  78.00 GB

Performance Gain:
  Speedup:      3.33x
  Time Saved:   210.00s
```

---

## STEP 11: Download Videos to Your Local Machine

### 11.1 From Your Local Terminal (NOT SSH session)

```bash
# Download a single video
scp -P <your-port> \
    root@<your-pod-id>.proxy.runpod.net:/workspace/Wan2.2/video1.mp4 \
    ~/Downloads/

# Download all videos
scp -P <your-port> \
    root@<your-pod-id>.proxy.runpod.net:/workspace/Wan2.2/*.mp4 \
    ~/Downloads/
```

### 11.2 Alternative: Use RunPod File Browser

1. In RunPod web interface, click **"Connect"** → **"HTTP Service [Port 8888]"**
2. Navigate to `/workspace/Wan2.2/`
3. Download videos directly through browser

---

## 📊 Quick Reference: Path Structure

```
/workspace/Wan2.2/                           # Main project directory
├── checkpoints/                             # Downloaded models
│   └── Wan2.2-T2V-A14B/                    # T2V checkpoint (~56GB)
│       ├── low_noise_model/
│       ├── high_noise_model/
│       ├── models_t5_umt5-xxl-enc-bf16.pth
│       ├── Wan2.1_VAE.pth
│       └── google/
├── generate_optimized.py                    # Optimized generation script
├── wan/
│   └── text2video_optimized.py             # Optimized pipeline
├── benchmark_t2v.py                         # Benchmark tool
├── *.mp4                                    # Generated videos
└── README_OPTIMIZATION.md                   # Documentation
```

---

## 🎯 Common Commands (Copy-Paste Ready)

### Set Checkpoint Path
```bash
export CKPT_DIR=/workspace/Wan2.2/checkpoints/Wan2.2-T2V-A14B
```

### Quick Test (17 frames)
```bash
python generate_optimized.py --ckpt_dir $CKPT_DIR --prompt "test" --frame_num 17 --no-offload
```

### Full Generation (81 frames, max performance)
```bash
python generate_optimized.py \
    --ckpt_dir $CKPT_DIR \
    --prompt "Your amazing prompt here" \
    --frame_num 81 \
    --no-offload \
    --compile_mode max-autotune
```

### Benchmark
```bash
python benchmark_t2v.py --ckpt_dir $CKPT_DIR --mode both --no-offload
```

### Monitor GPU
```bash
watch -n 1 nvidia-smi
```

---

## 🔧 Troubleshooting

### Problem: "CUDA out of memory"
**Solution:**
```bash
# Enable offloading
python generate_optimized.py --ckpt_dir $CKPT_DIR --prompt "..." --offload_model True
```

### Problem: "No module named 'wan'"
**Solution:**
```bash
cd /workspace/Wan2.2
python generate_optimized.py ...
```

### Problem: "Checkpoint directory not found"
**Solution:**
```bash
# Verify path
ls -la $CKPT_DIR
# If empty, re-download (see STEP 6.2)
```

### Problem: Flash Attention installation failed
**Solution:**
```bash
pip install flash_attn --no-build-isolation
```

### Problem: Download is very slow
**Solution:**
```bash
# Use ModelScope (faster in some regions)
pip install modelscope
modelscope download Wan-AI/Wan2.2-T2V-A14B --local_dir ./checkpoints/Wan2.2-T2V-A14B
```

### Problem: "Connection timeout" to RunPod
**Solution:**
```bash
# Check pod is running in RunPod dashboard
# Try reconnecting
ssh root@<pod-id>.proxy.runpod.net -p <port>
```

---

## 💰 Cost Estimates (RunPod H200)

| Configuration | Cost/hour | 81 frames (optimized) | 81 frames (original) |
|--------------|-----------|----------------------|---------------------|
| H200 80GB On-Demand | ~$4-5/hr | 90s = $0.13 | 300s = $0.42 |
| H200 SXM Spot | ~$2-3/hr | 90s = $0.08 | 300s = $0.25 |

**With optimization, you save ~70% on compute costs per video!**

---

## 📈 Expected Timeline

| Task | Time |
|------|------|
| Launch RunPod instance | 2-5 min |
| SSH connect & setup | 2-3 min |
| Clone repo | 1 min |
| Install dependencies | 5-10 min |
| Download checkpoint | 10-30 min |
| First test generation | 1-2 min |
| **Total setup** | **~25-50 min** |
| | |
| Full generation (first) | ~120s |
| Full generation (after warmup) | ~85s |
| Benchmark (both modes, 3 runs) | ~20 min |

---

## ✅ Final Checklist

Before generating your first video:

- [ ] H200 instance running on RunPod
- [ ] Connected via SSH
- [ ] PyTorch 2.4+ installed
- [ ] Wan2.2 repo cloned
- [ ] Dependencies installed (flash_attn)
- [ ] Checkpoint downloaded (~56GB)
- [ ] `$CKPT_DIR` environment variable set
- [ ] Quick test successful (17 frames)
- [ ] Ready for full generation!

---

## 🎉 You're All Set!

Generate your first optimized video:

```bash
cd /workspace/Wan2.2

python generate_optimized.py \
    --ckpt_dir $CKPT_DIR \
    --prompt "Two anthropomorphic cats in boxing gear fighting intensely on a spotlighted stage" \
    --frame_num 81 \
    --no-offload \
    --compile_mode max-autotune \
    --seed 42 \
    --save_file my_first_video.mp4
```

**Enjoy 3.5x faster video generation on H200!** 🚀

---

## 📞 Need Help?

- Check `OPTIMIZATION_GUIDE.md` for detailed configuration options
- Check `README_OPTIMIZATION.md` for troubleshooting
- Monitor GPU: `watch -n 1 nvidia-smi`
- Check disk space: `df -h`
- Check process: `htop`

---

*Last updated: 2025-10-25*  
*Tested on: RunPod H200 80GB/141GB with PyTorch 2.4+*

