# U10 Training Fixes

This document describes the fixes applied to resolve training issues with U10_train.py.

## Problem 1: KeyError: 'bases' during VecNormalize reset

### Root Cause
The observation dict returned by `_get_observation()` was missing the `'objective_weights'` key that was declared in `observation_space`. The key was added **after** calling `_get_observation()` in both `reset()` and `step()`, but this caused issues with the VecNormalize wrapper which validates observation structure during reset.

### Solution
1. Added `'objective_weights'` directly to the dict returned by `_get_observation()`
2. Removed redundant assignments in `reset()` and `step()`
3. Added validation in `_get_observation()` to catch future key mismatches early

### Files Changed
- `UAV_ENVIRONMENT_10.py`:
  - Modified `_get_observation()` to include `'objective_weights'`
  - Added assertion to validate observation keys match observation_space keys
  - Removed redundant `obs['objective_weights'] = ...` lines in `reset()` and `step()`

## Problem 2: Episode Length Confusion

### Root Cause
The EventDrivenSingleUAVWrapper calls the base environment's `step()` multiple times per wrapper step:
- Once for each decision (after applying the action)
- Multiple times when skipping forward to find the next decision point

This means:
- Base environment runs for exactly 192 steps (= (22-6) * 12)
- Wrapper step count is much lower (~90-140 steps per episode)
- SB3's `ep_len_mean` metric reports wrapper steps, not env steps

### Solution
1. Added `total_env_steps` counter to track actual environment time steps
2. Modified `step()` to report `env_steps` (steps taken in this call) and `total_env_steps` (cumulative) in the info dict
3. Updated observation_space to properly include `'current_drone_id'` key
4. Added validation to ensure observation keys match observation_space
5. Added documentation explaining the episode length behavior

### Files Changed
- `U10_event_driven_single_uav_wrapper.py`:
  - Added `total_env_steps` counter in `__init__()` and `reset()`
  - Modified `step()` to track and report env_steps
  - Modified `_skip_to_next_decision()` to return steps_taken
  - Updated observation_space definition to include `'current_drone_id'`
  - Added validation in `_get_current_observation()`
  - Enhanced docstrings to explain episode length behavior

## Verification

### Test Results
All tests passed:
```
Base Environment:
  ✓ Observation keys match observation_space
  ✓ Episode length = 192 (as expected)

Event-Driven Wrapper:
  ✓ Observation keys match observation_space (including current_drone_id)
  ✓ Environment steps tracked correctly (192)
  ✓ Wrapper steps (~142) < env steps (192) - expected due to skipping
```

### Training Test
Successfully ran training with:
```bash
python U10_train.py --total-steps 100 --num-drones 3 --candidate-k 5 \
  --mopso-n-particles 5 --mopso-n-iterations 2
```

Results:
- No KeyError during training
- Environment completes 192 steps per episode (shown in logs)
- SB3 reports `ep_len_mean = 91.4` (wrapper steps, as expected)

## Episode Length Metrics Explained

| Metric | Value | Description |
|--------|-------|-------------|
| Base env steps | 192 | Actual time steps in environment = (22-6) * 12 |
| Wrapper steps | ~90-140 | Number of decisions made by RL agent |
| SB3 ep_len_mean | ~90-140 | Reports wrapper steps (what SB3 sees) |
| info['total_env_steps'] | 192 | Actual environment time steps (available in info) |

**Why the difference?**
The wrapper is event-driven: it only calls the RL agent when a drone needs a decision. When no drones need decisions, it automatically advances the environment by calling `env.step()` multiple times until a decision point is found. This reduces the number of decisions the agent needs to make while still advancing the simulation time.

## Usage

To track actual environment time steps during training, access `info['total_env_steps']`:

```python
obs, reward, done, truncated, info = env.step(action)
env_steps = info['total_env_steps']  # Actual environment time steps
```

For logging, you can create a custom callback that logs both metrics to TensorBoard or your preferred logging system.
