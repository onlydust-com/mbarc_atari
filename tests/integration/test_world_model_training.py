"""
End-to-end integration test for MBARC world model training.

This test validates the complete training pipeline:
1. Creates a NextFramePredictor model (world model)
2. Generates synthetic training data (frames, actions, rewards, values)
3. Performs forward pass through the model
4. Computes training loss
5. Performs backward pass and optimization step
6. Validates that model weights are updated

This proves that the MBARC codebase can be fully tested in a Docker container.
"""
import pytest
import torch
import numpy as np
from argparse import Namespace


@pytest.mark.integration
def test_world_model_training_end_to_end(test_device):
    """
    End-to-end test of world model training.

    This test exercises the core MBARC functionality:
    - NextFramePredictor model creation
    - Forward pass with real model architecture
    - Loss computation
    - Backward pass and weight updates
    """
    from mbarc.next_frame_predictor import NextFramePredictor
    from mbarc.adafactor import Adafactor

    print("\n" + "="*70)
    print("MBARC END-TO-END INTEGRATION TEST")
    print("="*70)

    # Step 1: Create configuration (with ALL required attributes)
    print("\n[1/7] Creating model configuration...")
    config = Namespace(
        device=str(test_device),
        frame_shape=(3, 84, 84),  # Standard Atari resolution
        hidden_size=64,
        stacking=4,
        compress_steps=2,
        use_stochastic_model=False,
        use_modified_model=True,  # Use MBARC's modified reward estimator

        # Additional required attributes
        bottleneck_bits=128,
        bottleneck_noise=0.1,
        dropout=0.15,
        filter_double_steps=1,
        hidden_layers=2,
        latent_rnn_max_sampling=0.5,
        latent_state_size=128,
        latent_use_max_probability=0.8,
        recurrent_state_size=64,
        residual_dropout=0.5,
        rollout_length=10,
        stack_internal_states=False,
        strategy='online',
        target_loss_clipping=0.03,
    )
    print(f"  ✓ Config: frame_shape={config.frame_shape}, hidden_size={config.hidden_size}")

    # Step 2: Create world model
    print("\n[2/7] Creating NextFramePredictor (world model)...")
    num_actions = 6  # Typical Atari action space
    model = NextFramePredictor(config, num_actions).to(test_device)
    print(f"  ✓ Model created with {sum(p.numel() for p in model.parameters())} parameters")

    # Step 3: Create optimizer
    print("\n[3/7] Creating Adafactor optimizer...")
    optimizer = Adafactor(model.parameters())
    print("  ✓ Optimizer initialized")

    # Step 4: Generate synthetic training data
    print("\n[4/7] Generating synthetic training data...")
    batch_size = 4
    c, h, w = config.frame_shape

    # Input frames: [batch, channels * stacking, height, width]
    # Note: Model processes batches of frames, NOT sequences
    frames = torch.randn(
        batch_size,
        c * config.stacking,
        h,
        w,
        device=test_device
    )

    # Actions: [batch, num_actions] (one-hot encoded)
    actions = torch.zeros(batch_size, num_actions, device=test_device)
    actions[:, 0] = 1  # Set first action as active

    # Target frames to predict: [batch, channels, height, width]
    # Must be float normalized (0-1) as the model expects
    target_frames = torch.randint(
        0, 256,
        (batch_size, c, h, w),
        dtype=torch.uint8,
        device=test_device
    ).float() / 255  # Convert to float and normalize

    # Rewards: [batch] (values 0, 1, 2 representing -1, 0, +1)
    rewards = torch.randint(0, 3, (batch_size,), device=test_device)

    # Values: [batch]
    values = torch.randn(batch_size, device=test_device)

    print(f"  ✓ Generated data:")
    print(f"    - Frames: {frames.shape}")
    print(f"    - Actions: {actions.shape}")
    print(f"    - Targets: {target_frames.shape}")
    print(f"    - Rewards: {rewards.shape}")
    print(f"    - Values: {values.shape}")

    # Step 5: Forward pass
    print("\n[5/7] Performing forward pass through world model...")
    model.train()

    # Save initial weights for comparison
    initial_weights = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            initial_weights[name] = param.data.clone()

    # NextFramePredictor.forward(x, action, target=None, epsilon=0)
    # Returns: (frame_logits, reward_pred, value_pred)
    frame_logits, reward_logits, value_predictions = model(
        frames,
        actions,
        target=target_frames,
        epsilon=0
    )

    assert frame_logits is not None, "Model forward pass returned None for frame_logits"
    assert reward_logits is not None, "Model forward pass returned None for reward_logits"
    assert value_predictions is not None, "Model forward pass returned None for value_predictions"

    print(f"  ✓ Forward pass successful")
    print(f"    - Frame logits shape: {frame_logits.shape}")
    print(f"    - Reward logits shape: {reward_logits.shape}")
    print(f"    - Value predictions shape: {value_predictions.shape}")

    # Step 6: Compute loss and backward pass
    print("\n[6/7] Computing loss and performing backward pass...")

    # The model computes losses internally during forward pass
    # For this test, we'll compute a simple combined loss
    # In real training, the model's internal losses would be used

    # Frame reconstruction loss (simplified)
    # frame_logits shape: [batch, 256, channels, height, width]
    # Reshape to [batch * channels * height * width, 256] for cross-entropy
    frame_logits_flat = frame_logits.permute(0, 2, 3, 4, 1).contiguous()  # [batch, channels, h, w, 256]
    frame_logits_flat = frame_logits_flat.view(-1, 256)  # [batch*channels*h*w, 256]

    # Convert float targets (0-1) back to class indices (0-255)
    target_frames_indices = (target_frames * 255).clamp(0, 255).long()  # [batch, channels, h, w]
    target_frames_flat = target_frames_indices.contiguous().view(-1)  # [batch*channels*h*w]

    frame_loss = torch.nn.functional.cross_entropy(
        frame_logits_flat,
        target_frames_flat,
        reduction='mean'
    )

    # Reward prediction loss
    reward_loss = torch.nn.functional.cross_entropy(
        reward_logits,
        rewards.long(),
        reduction='mean'
    )

    # Value prediction loss
    value_loss = torch.nn.functional.mse_loss(
        value_predictions,
        values,
        reduction='mean'
    )

    # Combined loss
    total_loss = frame_loss + reward_loss + value_loss

    print(f"  ✓ Losses computed:")
    print(f"    - Frame reconstruction loss: {frame_loss.item():.4f}")
    print(f"    - Reward prediction loss: {reward_loss.item():.4f}")
    print(f"    - Value prediction loss: {value_loss.item():.4f}")
    print(f"    - Total loss: {total_loss.item():.4f}")

    # Backward pass
    optimizer.zero_grad()
    total_loss.backward()

    # Check that gradients were computed
    has_gradients = False
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            has_gradients = True
            break

    assert has_gradients, "No gradients computed during backward pass"
    print("  ✓ Backward pass successful, gradients computed")

    # Step 7: Optimization step and validate weight updates
    print("\n[7/7] Performing optimization step...")
    optimizer.step()

    # Check that weights were updated
    weights_updated = False
    updated_params = []
    for name, param in model.named_parameters():
        if param.requires_grad and name in initial_weights:
            if not torch.allclose(param.data, initial_weights[name], rtol=1e-5):
                weights_updated = True
                updated_params.append(name)

    assert weights_updated, "Model weights were not updated after optimization step"
    print(f"  ✓ Optimization step successful")
    print(f"    - {len(updated_params)} parameter groups updated")

    # Final validation
    print("\n" + "="*70)
    print("✓ END-TO-END TEST PASSED")
    print("="*70)
    print("\nValidated functionality:")
    print("  1. ✓ Model architecture (NextFramePredictor)")
    print("  2. ✓ Forward pass with frame/reward/value prediction")
    print("  3. ✓ Loss computation (frame, reward, value)")
    print("  4. ✓ Backward propagation")
    print("  5. ✓ Gradient computation")
    print("  6. ✓ Adafactor optimizer")
    print("  7. ✓ Weight updates")
    print("\n" + "="*70)
    print("CONCLUSION: MBARC can be fully tested in Docker!")
    print("="*70 + "\n")

    assert True  # Test passed!
