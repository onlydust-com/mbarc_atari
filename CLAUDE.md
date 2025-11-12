# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is the official implementation of **Model-Based Aware Reward Classification (MBARC)** for the Atari 100k benchmark. MBARC is a model-based reinforcement learning approach that combines world model learning with specialized reward prediction strategies.

The system learns a world model that predicts next frames and rewards from the real environment, then trains a PPO agent in the simulated environment produced by the world model. This allows for sample-efficient learning on Atari games with only 100k environment steps.

## Requirements

- Python 3.7
- CUDA 10.2 (optional, can run on CPU)
- Tested on Ubuntu 20.04.1

## Common Commands

### Installation
```bash
pip install torch==1.7.1 gym==0.15.7 gym[atari] opencv-python==4.4.0.42 numpy==1.16.4 tqdm
git clone https://github.com/openai/baselines.git
pip install -e baselines

# Optional: wandb for experiment tracking
pip install wandb
```

### Running the program
```bash
# Basic usage (CUDA enabled by default)
python -m mbarc

# Run on CPU
python -m mbarc --device cpu

# Run with custom environment (default is Freeway)
python -m mbarc --env-name Pong

# Run with wandb tracking
python -m mbarc --use-wandb

# Render during training
python -m mbarc --render-training

# Use different training strategy
python -m mbarc --strategy mbarc  # Options: online, class_balanced, square_root, progressively_balanced, mbarc
```

### Key flags
- `--agents N`: Number of parallel environments (default: 16)
- `--device DEVICE`: PyTorch device (default: cuda)
- `--env-name NAME`: Atari game name without suffixes (default: Freeway)
- `--epochs N`: Training epochs (default: 15)
- `--strategy STRATEGY`: Training strategy for reward prediction
- `--save-models`: Save trained models to `models/` directory
- `--load-models`: Load models from `models/` directory

## Architecture

### Core Training Loop (SimPLe class in `mbarc/__main__.py`)

The training follows this cycle for each epoch:
1. **Random exploration** (epoch 0 only): 6400 random steps to seed the replay buffer
2. **Collect interactions**: 6400 steps using the current policy with temperature sampling
3. **Train world model**: Learns to predict next frames, rewards, and values
4. **Train reward model**: Separately trains reward prediction head with class balancing
5. **Train agent in simulation**: PPO agent trains on simulated rollouts from the world model

### Key Components

#### World Model (`mbarc/next_frame_predictor.py:NextFramePredictor`)
- **Encoder-decoder architecture** with downscaling/upscaling convolutions
- Predicts: next frame (256-way classification per pixel), reward (3-way classification), and value (regression)
- **Stochastic latent variables** (optional): `StochasticModel` adds stochasticity via discrete latent codes predicted by an LSTM
- **Action injection**: Actions are injected at multiple layers via `ActionInjector`
- **Scheduled sampling**: During training, epsilon controls whether to feed ground-truth or predicted frames
- **Timing signals**: Position encodings added at each layer

#### Reward Prediction Strategies (`mbarc/trainer.py:train_reward_model`)
The key innovation of MBARC is how it handles class imbalance in rewards:
- `online`: Train reward head jointly with world model
- `class_balanced`: Sample inversely proportional to class frequency
- `square_root`: Sample inversely proportional to sqrt of class frequency
- `progressively_balanced`: Gradually shift from uniform to class-balanced sampling
- `mbarc`: Classify rewards into 3 categories (forward frames before reward, central frames of reward, other reward frames) and sample according to `--mbarc-dist` (default: 0.35, 0.05, 0.6)

#### Simulated Environment (`mbarc/simulated_env.py:SimulatedEnv`)
- Wraps the world model as a Gym environment
- Steps are computed by calling the world model's forward pass
- Environment is "restarted" with real frames from the replay buffer before each simulated rollout

#### Real Environment Wrapper (`atari_utils/atari_utils/envs.py:VecRecorderWrapper`)
- Records all interactions in a replay buffer with format: `[state, action, reward, next_state, done, value]`
- States are uint8 tensors (stacked frames)
- Actions are one-hot encoded
- Rewards are clipped to {-1, 0, +1} then shifted to {0, 1, 2}
- Values are computed retroactively when episode ends

#### PPO Agent (`atari_utils/atari_utils/ppo_wrapper.py:PPO`)
- Standard PPO implementation wrapping `a2c_ppo_acktr` library
- Switches between real and simulated environments via `set_env()`
- Temperature sampling wrapper (`SampleWithTemperature`) used during data collection

### Training Details

#### World Model Training (`mbarc/trainer.py:train_world_model`)
- Samples random rollouts of length `--rollout-length` (default: 50) from buffer
- Applies scheduled sampling in epoch 0 to gradually reduce teacher forcing
- Losses: frame reconstruction (cross-entropy with clipping), value (MSE), reward (cross-entropy or CBF), LSTM loss (for stochastic model)
- Optimizer: Adafactor

#### Agent Training in Simulation (`mbarc/__main__.py:train_agent_sim_env`)
- Simulated rollouts start from real frames sampled from buffer
- Number of training steps increases in epochs 7, 11, and 14 (z=2, z=2, z=3)
- One agent always starts from the first small rollout of the epoch (if `--simulation-flip-first-random-for-beginning`)

### Module Structure

```
mbarc/
├── __main__.py          # Entry point, SimPLe training loop
├── trainer.py           # World model and reward model training
├── next_frame_predictor.py  # World model architecture
├── simulated_env.py     # Simulated environment wrapper
├── subproc_vec_env.py   # Vectorized simulated environments
├── utils.py             # Helper functions and custom layers
├── focal_loss.py        # Class-balanced focal loss
└── adafactor.py         # Adafactor optimizer

atari_utils/atari_utils/
├── envs.py              # Real environment wrappers and VecRecorderWrapper
├── ppo_wrapper.py       # PPO agent wrapper
├── evaluation.py        # Evaluation utilities
├── policy_wrappers.py   # Policy sampling wrappers
├── logger.py            # WandB logger
└── utils.py             # Utility functions

a2c_ppo_acktr/a2c_ppo_acktr/
├── policy.py            # Actor-critic policy network
├── ppo.py               # PPO algorithm implementation
├── rollout_storage.py   # Rollout buffer for PPO
└── distributions.py     # Action distributions
```

### Important Implementation Details

- **OpenCV threading**: `cv2.setNumThreads(0)` is set to avoid PyTorch DataLoader issues
- **PyTorch multiprocessing**: Uses `file_system` sharing strategy
- **Frame standardization**: Input frames are normalized and have median noise added during world model training
- **Reward encoding**: Real rewards are clipped to {-1, 0, 1} then shifted to uint8 {0, 1, 2} for storage
- **Buffer storage**: All buffer elements use uint8 for memory efficiency except values (float32)
- **Internal states** (optional with `--stack-internal-states`): Recurrent states can be stacked with frames, but this disables reward model training and evaluation
- **Modified reward estimator**: `--use-modified-model` (default True) uses MBARC's proposed reward head instead of SimPLe's original design
