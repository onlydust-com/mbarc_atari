"""
Pytest configuration and shared fixtures for MBARC testing.
"""
import os
from argparse import Namespace

import pytest
import torch


# Register ALE environments at module import
try:
    import gymnasium as gym
    import ale_py
    gym.register_envs(ale_py)
except ImportError:
    pass  # ALE not available, skip Atari tests


@pytest.fixture(scope="session")
def test_device():
    """Get the device to use for testing (prefer CPU for CI)."""
    device = os.environ.get("DEVICE", "cpu")
    return torch.device(device)


@pytest.fixture(scope="session")
def minimal_config(test_device):
    """
    Create a minimal configuration for testing.

    This configuration uses minimal parameters to speed up tests while
    still exercising the core functionality.
    """
    return Namespace(
        # Environment settings
        env_name="Pong",  # Simple Atari game
        device=str(test_device),
        frame_shape=(3, 84, 84),  # Standard Atari resolution
        noop_max=1,  # Minimal NOOP actions

        # Training settings - minimal for speed
        agents=1,  # Single agent for faster testing
        batch_size=2,  # Minimal batch size
        epochs=1,  # Just one epoch for PoC
        rollout_length=10,  # Short rollouts (vs 50 in production)

        # Model settings
        hidden_size=32,  # Smaller than production (96)
        stacking=4,  # Standard frame stacking
        compress_steps=2,  # Compression steps for model
        use_stochastic_model=False,  # Disable for simplicity/speed
        use_modified_model=True,  # Use MBARC's modified reward estimator

        # PPO settings
        ppo_gamma=0.99,
        ppo_lr=1e-4,
        use_ppo_lr_decay=False,

        # Training strategy
        strategy="online",  # Simplest strategy for testing
        use_cbf_loss=False,  # Disable class-balanced focal loss

        # Input preprocessing
        input_noise=0.0,  # No noise for deterministic testing

        # Logging and saving
        use_wandb=False,  # Disable wandb for tests
        render_training=False,  # No rendering in headless environment
        save_models=False,  # Don't save during tests
        load_models=False,  # Don't load during tests

        # Simulation settings
        simulation_flip_first_random_for_beginning=False,

        # Experiment naming
        experiment_name="test_run",
    )


@pytest.fixture
def temp_model_dir(tmp_path):
    """Create a temporary directory for model saving/loading."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    return model_dir


@pytest.fixture(autouse=True)
def set_random_seeds():
    """Set random seeds for reproducibility in tests."""
    import numpy as np
    import random

    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Ensure deterministic behavior in PyTorch
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
