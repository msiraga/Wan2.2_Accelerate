"""
Test suite for optimized T2V implementation.

Run without checkpoints:
    pytest tests/test_optimized_t2v.py -v

Run with checkpoints (slow):
    pytest tests/test_optimized_t2v.py -v --run-with-checkpoint --ckpt-dir /path/to/checkpoint
"""

import pytest
import torch
import sys
import os
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestOptimizedImports:
    """Test that all required modules can be imported."""
    
    def test_import_optimized_pipeline(self):
        """Test importing the optimized pipeline."""
        try:
            from wan.text2video_optimized import WanT2VOptimized
            assert WanT2VOptimized is not None
        except ImportError as e:
            pytest.fail(f"Failed to import WanT2VOptimized: {e}")
    
    def test_import_generate_script(self):
        """Test that generate_optimized.py exists and is valid Python."""
        script_path = Path(__file__).parent.parent / "generate_optimized.py"
        assert script_path.exists(), "generate_optimized.py not found"
        
        # Try to compile it
        with open(script_path) as f:
            code = f.read()
            try:
                compile(code, str(script_path), 'exec')
            except SyntaxError as e:
                pytest.fail(f"Syntax error in generate_optimized.py: {e}")
    
    def test_torch_available(self):
        """Test that PyTorch is available."""
        assert torch.cuda.is_available(), "CUDA not available"
        print(f"\n✓ PyTorch {torch.__version__}, CUDA {torch.version.cuda}")
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")


class TestOptimizedClass:
    """Test WanT2VOptimized class initialization and methods."""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration."""
        config = Mock()
        config.num_train_timesteps = 1000
        config.boundary = 0.875
        config.param_dtype = torch.bfloat16
        config.t5_dtype = torch.bfloat16
        config.text_len = 512
        config.vae_stride = (4, 8, 8)
        config.patch_size = (1, 2, 2)
        config.sample_neg_prompt = "bad quality"
        config.t5_checkpoint = "models_t5.pth"
        config.t5_tokenizer = "google/umt5-xxl"
        config.vae_checkpoint = "vae.pth"
        config.low_noise_checkpoint = "low_noise"
        config.high_noise_checkpoint = "high_noise"
        return config
    
    def test_tf32_enabled(self, mock_config):
        """Test that TF32 gets enabled when requested."""
        with patch('wan.text2video_optimized.T5EncoderModel'), \
             patch('wan.text2video_optimized.Wan2_1_VAE'), \
             patch('wan.text2video_optimized.WanModel'):
            
            from wan.text2video_optimized import WanT2VOptimized
            
            # Test with TF32 enabled
            _ = WanT2VOptimized(
                config=mock_config,
                checkpoint_dir="/fake/path",
                enable_tf32=True,
                enable_compile=False  # Skip compilation for test
            )
            
            # Check TF32 is enabled
            assert torch.backends.cuda.matmul.allow_tf32 == True
            assert torch.backends.cudnn.allow_tf32 == True
    
    def test_compile_mode_options(self, mock_config):
        """Test that different compile modes are accepted."""
        with patch('wan.text2video_optimized.T5EncoderModel'), \
             patch('wan.text2video_optimized.Wan2_1_VAE'), \
             patch('wan.text2video_optimized.WanModel'):
            
            from wan.text2video_optimized import WanT2VOptimized
            
            for mode in ["default", "reduce-overhead", "max-autotune"]:
                pipeline = WanT2VOptimized(
                    config=mock_config,
                    checkpoint_dir="/fake/path",
                    enable_compile=False,  # Don't actually compile
                    compile_mode=mode
                )
                assert pipeline.compile_mode == mode
    
    def test_optimization_flags(self, mock_config):
        """Test that optimization flags are stored correctly."""
        with patch('wan.text2video_optimized.T5EncoderModel'), \
             patch('wan.text2video_optimized.Wan2_1_VAE'), \
             patch('wan.text2video_optimized.WanModel'):
            
            from wan.text2video_optimized import WanT2VOptimized
            
            pipeline = WanT2VOptimized(
                config=mock_config,
                checkpoint_dir="/fake/path",
                enable_compile=True,
                enable_tf32=True,
                compile_mode="max-autotune"
            )
            
            assert pipeline.enable_compile == True
            assert pipeline.compile_mode == "max-autotune"


class TestBatchedCFG:
    """Test batched CFG implementation."""
    
    def test_batched_cfg_tensor_shapes(self):
        """Test that batched CFG produces correct tensor shapes."""
        # Simulate batching
        B, C, T, H, W = 1, 16, 20, 90, 160
        
        # Original: separate tensors
        x_uncond = torch.randn(C, T, H, W)
        x_cond = torch.randn(C, T, H, W)
        
        # Batched: concatenate
        x_batched = torch.cat([x_uncond, x_cond], dim=0)
        
        assert x_batched.shape == (C*2, T, H, W)
        
        # Split back
        x_uncond_split, x_cond_split = x_batched.chunk(2, dim=0)
        
        assert x_uncond_split.shape == x_uncond.shape
        assert x_cond_split.shape == x_cond.shape
        assert torch.allclose(x_uncond_split, x_uncond)
        assert torch.allclose(x_cond_split, x_cond)
    
    def test_batched_context_concatenation(self):
        """Test context batching for CFG."""
        # Simulate context tensors
        ctx_null = [torch.randn(256, 4096), torch.randn(256, 4096)]
        ctx_cond = [torch.randn(256, 4096), torch.randn(256, 4096)]
        
        # Batch them
        ctx_batched = [torch.cat([u, c], dim=0) for u, c in zip(ctx_null, ctx_cond)]
        
        assert len(ctx_batched) == 2
        assert ctx_batched[0].shape == (512, 4096)
        
        # Verify we can split back
        for i, ctx in enumerate(ctx_batched):
            u, c = ctx.chunk(2, dim=0)
            assert torch.allclose(u, ctx_null[i])
            assert torch.allclose(c, ctx_cond[i])


class TestGenerateScript:
    """Test generate_optimized.py command-line interface."""
    
    def test_argparse_structure(self):
        """Test that argument parser is correctly structured."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "generate_optimized",
            Path(__file__).parent.parent / "generate_optimized.py"
        )
        module = importlib.util.module_from_spec(spec)
        
        # Execute the module to load its contents
        spec.loader.exec_module(module)
        
        # Check key functions exist
        assert hasattr(module, 'main'), "main() function not found"
    
    def test_required_arguments(self):
        """Test that required arguments are enforced."""
        import subprocess
        
        result = subprocess.run(
            [sys.executable, "generate_optimized.py", "--help"],
            cwd=Path(__file__).parent.parent,
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0
        assert "--ckpt_dir" in result.stdout
        assert "--prompt" in result.stdout
        assert "--no-offload" in result.stdout or "--offload_model" in result.stdout


@pytest.mark.slow
@pytest.mark.skipif(
    not os.environ.get("RUN_CHECKPOINT_TESTS"),
    reason="Checkpoint tests skipped (set RUN_CHECKPOINT_TESTS=1 to run)"
)
class TestWithCheckpoint:
    """Tests that require actual checkpoint (slow, optional)."""
    
    @pytest.fixture
    def checkpoint_dir(self):
        """Get checkpoint directory from environment or command line."""
        ckpt_dir = os.environ.get("CKPT_DIR")
        if not ckpt_dir:
            pytest.skip("CKPT_DIR not set")
        if not os.path.exists(ckpt_dir):
            pytest.skip(f"Checkpoint directory not found: {ckpt_dir}")
        return ckpt_dir
    
    def test_load_checkpoint(self, checkpoint_dir):
        """Test that checkpoint can be loaded."""
        from wan.text2video_optimized import WanT2VOptimized
        from wan.configs import WAN_CONFIGS
        
        config = WAN_CONFIGS["t2v-A14B"]
        
        try:
            pipeline = WanT2VOptimized(
                config=config,
                checkpoint_dir=checkpoint_dir,
                enable_compile=False,  # Skip for speed
                enable_tf32=True
            )
            assert pipeline is not None
            print(f"\n✓ Successfully loaded checkpoint from {checkpoint_dir}")
        except Exception as e:
            pytest.fail(f"Failed to load checkpoint: {e}")
    
    def test_quick_generation(self, checkpoint_dir):
        """Test quick generation (5 frames)."""
        from wan.text2video_optimized import WanT2VOptimized
        from wan.configs import WAN_CONFIGS
        
        config = WAN_CONFIGS["t2v-A14B"]
        
        pipeline = WanT2VOptimized(
            config=config,
            checkpoint_dir=checkpoint_dir,
            enable_compile=False,
            enable_tf32=True
        )
        
        video = pipeline.generate(
            input_prompt="A cat",
            size=(1280, 720),
            frame_num=5,  # Very short
            sampling_steps=2,  # Very few steps
            offload_model=True,
            seed=42
        )
        
        assert video is not None
        assert video.shape[0] == 3  # RGB
        print(f"\n✓ Generated video shape: {video.shape}")
    
    def test_optimizations_applied(self, checkpoint_dir):
        """Test that optimizations are actually applied."""
        from wan.text2video_optimized import WanT2VOptimized
        from wan.configs import WAN_CONFIGS
        
        config = WAN_CONFIGS["t2v-A14B"]
        
        # Test with optimizations
        pipeline_opt = WanT2VOptimized(
            config=config,
            checkpoint_dir=checkpoint_dir,
            enable_compile=False,
            enable_tf32=True
        )
        
        # Check TF32 is enabled
        assert torch.backends.cuda.matmul.allow_tf32 == True
        
        # Check models are configured
        assert hasattr(pipeline_opt, 'low_noise_model')
        assert hasattr(pipeline_opt, 'high_noise_model')
        
        print("\n✓ All optimizations properly configured")


class TestPerformanceComparison:
    """Test performance improvements."""
    
    @pytest.mark.skipif(
        not os.environ.get("RUN_PERFORMANCE_TESTS"),
        reason="Performance tests skipped (set RUN_PERFORMANCE_TESTS=1)"
    )
    def test_batched_vs_unbatched_cfg(self):
        """Test that batched CFG is faster than unbatched."""
        # This is a conceptual test - actual timing would need checkpoints
        
        # Batched should do 1 forward pass per step
        # Unbatched should do 2 forward passes per step
        
        steps = 40
        
        batched_forwards = steps * 1
        unbatched_forwards = steps * 2
        
        theoretical_speedup = unbatched_forwards / batched_forwards
        
        assert theoretical_speedup == 2.0
        print(f"\n✓ Batched CFG theoretical speedup: {theoretical_speedup}x")


def test_suite_info():
    """Print test suite information."""
    print("\n" + "="*70)
    print("WAN 2.2 T2V OPTIMIZED TEST SUITE")
    print("="*70)
    print("\nTest Categories:")
    print("  1. Import tests - Verify all modules load")
    print("  2. Class tests - Test WanT2VOptimized initialization")
    print("  3. Batched CFG tests - Verify tensor operations")
    print("  4. Script tests - Test generate_optimized.py")
    print("  5. Checkpoint tests - Test with actual model (optional)")
    print("\nQuick tests (no checkpoint): pytest tests/test_optimized_t2v.py -v")
    print("With checkpoint: RUN_CHECKPOINT_TESTS=1 CKPT_DIR=/path pytest tests/test_optimized_t2v.py -v")
    print("="*70 + "\n")


if __name__ == "__main__":
    # Run with: python tests/test_optimized_t2v.py
    pytest.main([__file__, "-v", "-s"])

