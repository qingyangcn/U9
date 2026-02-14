# U9

## PPO Training with VecNormalize for Improved Stability

This repository contains PPO training scripts for a multi-objective drone delivery environment with VecNormalize integration to improve training stability.

### Key Features

- **Rule-Based Discrete Actions**: 5 interpretable rules for drone order selection
- **MOPSO Assignment**: Multi-Objective Particle Swarm Optimization for order assignment
- **VecNormalize**: Observation and reward normalization for improved critic learning

### VecNormalize Integration

VecNormalize has been added to address training instability issues where:
- `explained_variance` was near 0 (or slightly negative) at 102,400 timesteps
- `value_loss` remained around 1e4, indicating poor value function fitting

**Benefits:**
- Normalizes rewards to reduce scale and improve critic learning
- Normalizes observations for better feature scaling (excluding discrete 'weather' key)
- Maintains running statistics (mean, variance) for consistent normalization
- Test runs show `value_loss` dropping from ~14 to <1 within 128 steps

### Training

```bash
python U9_train.py --total-steps 200000 --seed 42 --num-drones 50 --obs-max-orders 400
```

**Common training options:**
```bash
# Custom learning rate and batch size
python U9_train.py --lr 3e-4 --batch-size 128 --n-steps 4096

# Enable random events and diagnostics
python U9_train.py --enable-random-events --enable-diagnostics --diagnostics-interval 100

# Custom fallback policy
python U9_train.py --fallback-policy first_valid
```

The training script automatically saves:
- Model checkpoints at regular intervals (default: every 10,000 steps)
- VecNormalize statistics with each checkpoint
- Final model and VecNormalize statistics

**Saved files structure:**
```
./models/u7_task/
├── ppo_u9_task_10000_steps.zip       # Model checkpoint
├── vecnormalize_10000_steps.pkl      # VecNormalize stats for checkpoint
├── ppo_u9_task_20000_steps.zip
├── vecnormalize_20000_steps.pkl
├── ...
├── ppo_u9_task_final.zip             # Final model
└── vecnormalize_final.pkl            # Final VecNormalize stats
```

### Evaluation and Model Loading

**IMPORTANT**: When loading a trained model for evaluation or continued training, you must also load the VecNormalize statistics:

#### Option 1: Using the provided evaluation script

```bash
python evaluate_example.py \
    --model-path ./models/u7_task/ppo_u9_task_final.zip \
    --vecnormalize-path ./models/u7_task/vecnormalize_final.pkl \
    --num-episodes 10
```

#### Option 2: Custom evaluation code

```python
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv, VecMonitor

# Create environment (same config as training)
env = DummyVecEnv([env_fn])
env = VecMonitor(env)

# Load VecNormalize statistics (REQUIRED)
env = VecNormalize.load("./models/u7_task/vecnormalize_final.pkl", env)

# For evaluation, disable training mode
env.training = False
env.norm_reward = False

# Load model
model = PPO.load("./models/u7_task/ppo_u9_task_final", env=env)

# Evaluate
obs = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
```

#### Continuing Training from Checkpoint

```python
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv, VecMonitor

# Create environment
env = DummyVecEnv([env_fn])
env = VecMonitor(env)

# Load VecNormalize statistics (keep training=True for continued training)
env = VecNormalize.load("./models/u7_task/vecnormalize_20000_steps.pkl", env)
env.training = True  # Keep normalization training enabled
env.norm_reward = True

# Load model
model = PPO.load("./models/u7_task/ppo_u9_task_20000_steps", env=env)

# Continue training
model.learn(total_timesteps=100000)
```

### Files

- `U9_train.py`: Main training script with VecNormalize integration
- `evaluate_example.py`: Example evaluation script demonstrating proper VecNormalize loading
- `UAV_ENVIRONMENT_9.py`: Drone delivery environment
- `U7_mopso_dispatcher.py`: MOPSO planner for order assignment

### Training Parameters

Key hyperparameters (defaults):
- Learning rate: 1e-4
- PPO steps: 2048
- Batch size: 64
- Epochs: 10
- Gamma: 0.99
- GAE Lambda: 0.95
- VecNormalize clip: 10.0 (both obs and reward)
- Checkpoint frequency: 10,000 steps

### Troubleshooting

**Issue**: Model performs poorly during evaluation
- **Solution**: Ensure you loaded VecNormalize statistics with `VecNormalize.load()`

**Issue**: "VecNormalize only supports Box observation spaces" error
- **Solution**: This has been fixed by specifying `norm_obs_keys` to exclude the 'weather' Discrete space

**Issue**: VecNormalize statistics file not found
- **Solution**: Check that both the model and VecNormalize files were saved together during training

### Performance Notes

With VecNormalize enabled, you should observe:
- `explained_variance` gradually increasing from near-0 to positive values (e.g., 0.3-0.9)
- `value_loss` decreasing from initial high values (~14) to lower values (<5, ideally <1)
- More stable training with reduced fluctuations in metrics

These improvements indicate that the value function is learning to predict returns more accurately.
