# U9 - UAV Delivery System with Layered Decision Architecture

A multi-objective UAV (drone) delivery optimization system with a layered decision-making architecture.

## Overview

The U9 system implements a hierarchical approach to UAV delivery optimization:
- **Upper Layer**: Generates candidate order sets for each drone using configurable strategies
- **Lower Layer**: Applies rule-based selection from candidates using reinforcement learning (PPO)
- **Event-Driven Interface**: Supports homogeneous policy training with action space independent of drone count

## Key Features

### 1. Layered Decision Architecture
- **Candidate Generation**: Upper layer filters orders into candidate sets
- **Rule Selection**: Lower layer selects from candidates using interpretable rules
- **Decoupled Action Space**: Action space is `Discrete(5)` regardless of drone count N

### 2. Candidate Generation Strategies
- `NearestCandidateGenerator`: Selects K nearest orders by pickup distance
- `EarliestDeadlineCandidateGenerator`: Selects K orders with earliest deadlines
- `MixedHeuristicCandidateGenerator`: Combines distance and deadline with configurable weights
- `PSOMOPSOCandidateGenerator`: Placeholder for PSO/MOPSO integration (future)

### 3. Rule-Based Order Selection
Five interpretable rules for order selection:
- **Rule 0 (CARGO_FIRST)**: Prioritize delivering picked-up orders
- **Rule 1 (ASSIGNED_EDF)**: Earliest deadline first from assigned orders
- **Rule 2 (READY_EDF)**: Earliest deadline first from ready orders
- **Rule 3 (NEAREST_PICKUP)**: Closest pickup location
- **Rule 4 (SLACK_PER_DISTANCE)**: Maximize slack/distance ratio

### 4. Event-Driven Wrapper
- Processes drones one at a time at decision points
- Automatically advances time when no decisions needed
- Supports homogeneous policy parameter sharing across drones

## Installation

```bash
# Clone repository
git clone https://github.com/qingyangcn/U9.git
cd U9

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Demo Mode (No Training Dependencies)

```bash
# Run demo with default settings
python scripts/train_event_driven_rule_ppo.py --mode demo

# Run demo with custom settings
python scripts/train_event_driven_rule_ppo.py --mode demo \
    --num-drones 5 \
    --candidate-strategy mixed \
    --candidate-k 10 \
    --demo-steps 200
```

### Training with PPO

```bash
# Train with default settings
python scripts/train_event_driven_rule_ppo.py --mode train --total-steps 50000

# Train with custom configuration
python scripts/train_event_driven_rule_ppo.py --mode train \
    --total-steps 100000 \
    --num-drones 10 \
    --candidate-strategy nearest \
    --candidate-k 15 \
    --learning-rate 0.0003
```

### Original Multi-Drone Training

```bash
# Original U9 training script (MultiDiscrete action space)
python U9_train.py --total-steps 200000 --seed 42 --num-drones 50
```

## Usage Guide

### Using Candidate-Based Filtering

```python
from UAV_ENVIRONMENT_9 import ThreeObjectiveDroneDeliveryEnv
from candidate_generator import MixedHeuristicCandidateGenerator

# Create environment with candidate filtering
env = ThreeObjectiveDroneDeliveryEnv(
    num_drones=5,
    candidate_fallback_enabled=True,  # Fallback to all orders if candidates empty
    candidate_update_interval=1,      # Update candidates every step
)

# Create and set candidate generator
generator = MixedHeuristicCandidateGenerator(
    candidate_k=20,
    distance_weight=0.6,
    deadline_weight=0.4
)
env.set_candidate_generator(generator)

# Use environment normally
obs, info = env.reset(seed=42)
```

### Using Event-Driven Wrapper

```python
from UAV_ENVIRONMENT_9 import ThreeObjectiveDroneDeliveryEnv
from candidate_generator import NearestCandidateGenerator
from wrappers import EventDrivenSingleUAVWrapper

# Create base environment
base_env = ThreeObjectiveDroneDeliveryEnv(
    num_drones=10,
    candidate_fallback_enabled=True,
)

# Set candidate generator
generator = NearestCandidateGenerator(candidate_k=15)
base_env.set_candidate_generator(generator)

# Wrap with event-driven interface
env = EventDrivenSingleUAVWrapper(
    base_env,
    max_skip_steps=10,      # Max steps to skip when waiting for decisions
    local_observation=False # Use global obs for now (will add local obs later)
)

# Action space is now Discrete(5) regardless of drone count
obs, info = env.reset(seed=42)
action = env.action_space.sample()  # Single rule_id (0-4)
obs, reward, terminated, truncated, info = env.step(action)

# Check statistics
stats = env.get_statistics()
print(f"Decisions made: {stats['total_decisions']}")
print(f"Steps skipped: {stats['total_skips']}")
```

### Custom Candidate Generator

```python
from candidate_generator import CandidateGenerator

class CustomCandidateGenerator(CandidateGenerator):
    def generate_candidates(self, env):
        """Generate candidates using custom logic."""
        candidates = {}
        
        for drone_id in range(env.num_drones):
            # Your custom logic here
            # Return list of order_ids for this drone
            candidates[drone_id] = [...]
        
        return candidates

# Use your custom generator
env.set_candidate_generator(CustomCandidateGenerator(candidate_k=20))
```

### Integrating PSO/MOPSO

To integrate Particle Swarm Optimization for candidate generation:

1. Implement the PSO/MOPSO algorithm in `PSOMOPSOCandidateGenerator.generate_candidates()`
2. The method receives the environment and should return `Dict[int, List[int]]`
3. Optimize objectives like distance, deadline urgency, load balance, etc.

```python
from candidate_generator import PSOMOPSOCandidateGenerator

class MyPSOGenerator(PSOMOPSOCandidateGenerator):
    def generate_candidates(self, env):
        # TODO: Implement PSO/MOPSO optimization
        # 1. Define particle representation (candidate assignments)
        # 2. Define fitness function (multi-objective)
        # 3. Run PSO iterations
        # 4. Return best candidate sets
        
        # For now, falls back to mixed heuristic
        return super().generate_candidates(env)
```

## Configuration Parameters

### Environment Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `candidate_fallback_enabled` | bool | True | Allow fallback to all orders if candidates empty |
| `candidate_update_interval` | int | 1 | Update candidates every N steps (0=only on reset) |

### Wrapper Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_skip_steps` | int | 10 | Max environment steps to skip when waiting for decisions |
| `local_observation` | bool | False | Use local observation instead of global (future) |

### Candidate Generator Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `candidate_k` | int | 20 | Number of candidates per drone |
| `distance_weight` | float | 0.5 | Weight for distance in mixed strategy |
| `deadline_weight` | float | 0.5 | Weight for deadline in mixed strategy |

## Testing

```bash
# Run all tests
python -m unittest discover tests

# Run specific test
python -m unittest tests.test_candidate_filtering
python -m unittest tests.test_event_driven_wrapper
```

## Architecture

### Directory Structure

```
U9/
├── UAV_ENVIRONMENT_9.py          # Main environment
├── candidate_generator.py        # Candidate generation strategies
├── wrappers/
│   ├── __init__.py
│   └── event_driven_single_uav_wrapper.py  # Event-driven wrapper
├── scripts/
│   └── train_event_driven_rule_ppo.py     # Training script
├── tests/
│   ├── test_candidate_filtering.py
│   └── test_event_driven_wrapper.py
├── U9_train.py                   # Original training script
└── requirements.txt

```

### Key Methods

**Environment Methods:**
- `set_candidate_generator(generator)`: Set external candidate generator
- `update_filtered_candidates()`: Update candidate sets
- `get_decision_drones()`: Get list of drones at decision points
- `apply_rule_to_drone(drone_id, rule_id)`: Apply rule to single drone

**Wrapper Methods:**
- `reset()`: Reset environment and initialize decision queue
- `step(action)`: Execute decision for current drone
- `get_statistics()`: Get wrapper statistics

## Backward Compatibility

The system maintains backward compatibility:
- Original `MultiDiscrete([5]*N)` action space still works
- Candidate filtering is optional (only active when generator is set)
- Fallback to original behavior when `candidate_fallback_enabled=True`

## Contributing

When adding new candidate generation strategies:
1. Inherit from `CandidateGenerator`
2. Implement `generate_candidates(env)` method
3. Add tests in `tests/test_candidate_filtering.py`
4. Update this README with usage examples

## License

[Specify license here]

## Citation

[Add citation information if applicable]