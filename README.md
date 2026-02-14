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
- Normalizes observations for better feature scaling
- Maintains running statistics (mean, variance) for consistent normalization

### Training

```bash
python U9_train.py --total-steps 200000 --seed 42 --num-drones 50 --obs-max-orders 400
```

The training script automatically saves:
- Model checkpoints at regular intervals
- VecNormalize statistics with each checkpoint
- Final model and VecNormalize statistics

### Evaluation and Model Loading

**IMPORTANT**: When loading a trained model for evaluation or continued training, you must also load the VecNormalize statistics:

```python
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

# Create environment
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

### Files

- `U9_train.py`: Main training script with VecNormalize integration
- `UAV_ENVIRONMENT_9.py`: Drone delivery environment
- `U7_mopso_dispatcher.py`: MOPSO planner for order assignment

### Training Parameters

Key hyperparameters:
- Learning rate: 1e-4
- PPO steps: 2048
- Batch size: 64
- Gamma: 0.99
- VecNormalize clip: 10.0 (both obs and reward)
