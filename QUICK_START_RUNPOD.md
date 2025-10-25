# ⚡ Quick Start: RunPod H200 Commands

**Copy-paste commands for fast setup**

---

## 🔌 1. Connect to RunPod

```bash
ssh root@<your-pod-id>.proxy.runpod.net -p <your-port>
```

---

## 🚀 2. One-Shot Setup Script

```bash
# Navigate to workspace
cd /workspace

# Clone repo
git clone https://github.com/Wan-Video/Wan2.2.git
cd Wan2.2

# Install dependencies
pip install -r requirements.txt
pip install flash_attn --no-build-isolation

# Download checkpoint (~56GB, takes 10-30 min)
mkdir -p checkpoints && cd checkpoints
pip install "huggingface_hub[cli,hf_transfer]"
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download \
    Wan-AI/Wan2.2-T2V-A14B \
    --local-dir ./Wan2.2-T2V-A14B

# Return to project root
cd /workspace/Wan2.2

# Set checkpoint path
export CKPT_DIR=/workspace/Wan2.2/checkpoints/Wan2.2-T2V-A14B
```

---

## 🎬 3. Generate Video (Maximum Performance)

```bash
python generate_optimized.py \
    --ckpt_dir $CKPT_DIR \
    --prompt "Two anthropomorphic cats in boxing gear fighting on stage" \
    --frame_num 81 \
    --no-offload \
    --compile_mode max-autotune \
    --seed 42
```

---

## 📊 4. Benchmark

```bash
python benchmark_t2v.py \
    --ckpt_dir $CKPT_DIR \
    --mode both \
    --num_runs 3 \
    --no-offload
```

---

## 📥 5. Download Videos (From Your Local Machine)

```bash
scp -P <port> root@<pod-id>.proxy.runpod.net:/workspace/Wan2.2/*.mp4 ~/Downloads/
```

---

## 🔍 Useful Commands

```bash
# Monitor GPU
watch -n 1 nvidia-smi

# Check disk space
df -h

# List generated videos
ls -lh *.mp4

# Check download progress
du -sh checkpoints/Wan2.2-T2V-A14B/

# View last 20 lines of output
tail -20 <output-file>
```

---

## 🎯 Expected Performance

- **Setup time**: 25-50 minutes
- **First generation**: ~120s (includes compilation)
- **Subsequent**: ~85s per video
- **VRAM usage**: ~78GB / 141GB
- **Speedup**: 3.3-3.5x vs baseline

---

For detailed instructions, see **RUNPOD_H200_SETUP.md**

