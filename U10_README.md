# U10 System Documentation

## Overview

U10 is an advanced multi-UAV delivery system that implements **Centralized Training, Decentralized Execution (CTDE)** for intelligent order assignment and delivery. The system uses a hierarchical decision-making architecture with event-driven execution.

## Architecture

### System Components

1. **Upper Layer: Candidate Generation (MOPSO)**
   - Generates K candidate orders for each drone using Multi-Objective Particle Swarm Optimization (MOPSO)
   - Considers multiple objectives: delivery time, energy efficiency, on-time rate
   - Only generates suggestions - does NOT commit orders (READY → ASSIGNED)
   - Refreshes periodically based on `candidate_update_interval`

2. **Lower Layer: Rule Selection (PPO + Rules)**
   - Uses Proximal Policy Optimization (PPO) to learn which rule to apply
   - 5 interpretable rules for order selection:
     - Rule 0: Select nearest order
     - Rule 1: Select order with earliest deadline
     - Rule 2: Select order with highest priority
     - Rule 3: Select order with best profit/distance ratio
     - Rule 4: Return to base (no order selection)
   - Actual order assignment (READY → ASSIGNED) happens via centralized arbitrator

3. **Centralized Arbitrator (Environment)**
   - Manages shared order pool atomically
   - Prevents race conditions when multiple drones select same order
   - Returns success/failure with reasons for each decision
   - Ensures state consistency across all drones

### CTDE (Centralized Training, Decentralized Execution)

**Centralized Training:**
- Single shared policy trained with experiences from all drones
- Action space: Discrete(5) - one rule_id per decision
- Observation: Local observation (drone_state, candidates, global_context)
- Training uses `SingleUAVTrainingWrapper` to sample decisions from different drones

**Decentralized Execution:**
- Each drone independently uses the same policy with its own local observation
- No central queue or controller needed during execution
- Drones only make decisions when at decision points (event-driven)
- System fast-forwards time when no decisions are needed

**Why CTDE?**
- **Scalability**: Action space doesn't grow with number of drones (always Discrete(5))
- **Generalization**: Same policy works for any number of drones
- **Efficiency**: Only act when needed (event-driven), not every time step
- **Robustness**: No single point of failure during execution

### Event-Driven Execution

The system uses an event-driven architecture where:

1. **Decision Points**: Drones reach decision points when:
   - Idle at base with available capacity
   - Completed a delivery and ready for next order
   - Reached merchant and ready to pick up

2. **Decision Process**:
   - System detects all drones at decision points
   - For each drone: extract local observation → call policy → submit to arbitrator
   - Arbitrator atomically handles order assignments
   - System advances one time step

3. **Fast-Forward**:
   - When no drones at decision points, system advances multiple steps
   - Stops when decision event occurs or episode ends
   - Avoids wasting computation on no-op steps

### Centralized Arbitration

While execution is decentralized, **order assignment must be centralized** to prevent conflicts:

**Why Centralized Arbitration?**
- Shared resource: Multiple drones compete for same orders
- Atomicity: Order status change (READY → ASSIGNED) must be atomic
- Consistency: Prevent double-assignment of orders

**How It Works:**
1. Drone submits `(drone_id, rule_id)` to arbitrator
2. Arbitrator checks:
   - Is drone at decision point?
   - Does rule select a valid order?
   - Is order still READY and unassigned?
   - Does drone have capacity?
3. If all checks pass: assign order to drone atomically
4. Return success/failure with reason

**Decision Tracking:**
- `last_decision_drone_id`: Which drone made last decision
- `last_decision_rule_id`: Which rule was applied
- `last_decision_success`: Whether decision succeeded
- `last_decision_failure_reason`: Why decision failed (if applicable)
  - Possible reasons:
    - `invalid_drone_id`: Invalid drone ID
    - `not_at_decision_point`: Drone not ready to decide
    - `no_order_selected`: Rule didn't select any order
    - `order_already_assigned`: Order taken by another drone
    - `drone_at_capacity`: Drone can't take more orders
    - `assignment_rejected`: Assignment failed for other reasons

## Training

### Training Modes

**1. Event-Driven Shared Policy (Default - CTDE)**
```bash
python U10_train.py \
    --training-mode event_driven_shared_policy \
    --total-steps 200000 \
    --num-drones 20 \
    --candidate-k 20 \
    --seed 42
```

Features:
- Trains single shared policy for all drones
- Action space: Discrete(5)
- Observation: Local (drone_state, candidates, global_context)
- Drone sampling: Random selection from drones at decision points
- Enables true decentralized execution

**2. Central Queue (Legacy)**
```bash
python U10_train.py \
    --training-mode central_queue \
    --total-steps 200000 \
    --num-drones 20
```

Features:
- Centralized queue-based decision making
- Action space: Discrete(5)
- Observation: Full observation with current_drone_id
- Processes drones one by one

### Training Parameters

**Environment:**
- `--num-drones`: Number of drones (default: 20)
- `--candidate-k`: Candidates per drone (default: 20)
- `--obs-max-orders`: Max orders in observation (default: 400)
- `--top-k-merchants`: Top K merchants (default: 100)

**MOPSO (Upper Layer):**
- `--mopso-n-particles`: Particle count (default: 30)
- `--mopso-n-iterations`: Iteration count (default: 10)
- `--mopso-max-orders`: Max orders for MOPSO (default: 200)

**PPO (Lower Layer):**
- `--lr`: Learning rate (default: 1e-4)
- `--n-steps`: Steps per rollout (default: 2048)
- `--batch-size`: Batch size (default: 64)
- `--gamma`: Discount factor (default: 0.99)

**Wrapper:**
- `--max-skip-steps`: Max fast-forward steps (default: 10)
- `--drone-sampling`: Sampling strategy (random/round_robin, default: random)

### Quick Test

```bash
# Quick training test (1000 steps)
python U10_train.py --total-steps 1000

# Quick test with fewer drones
python U10_train.py --total-steps 1000 --num-drones 5
```

## Deployment / Evaluation

### Using Decentralized Executor

```python
from UAV_ENVIRONMENT_10 import ThreeObjectiveDroneDeliveryEnv
from U10_candidate_generator import MOPSOCandidateGenerator
from U10_decentralized_execution import DecentralizedEventDrivenExecutor
from stable_baselines3 import PPO

# Create environment
env = ThreeObjectiveDroneDeliveryEnv(
    num_drones=20,
    num_candidates=20,
    # ... other params
)

# Set candidate generator
candidate_gen = MOPSOCandidateGenerator(candidate_k=20)
env.set_candidate_generator(candidate_gen)

# Load trained policy
model = PPO.load("./models/u10/ppo_u10_final.zip")

def policy_fn(local_obs):
    action, _ = model.predict(local_obs, deterministic=True)
    return int(action)

# Create executor
executor = DecentralizedEventDrivenExecutor(
    env=env,
    policy_fn=policy_fn,
    max_skip_steps=10,
    verbose=True
)

# Run episode
stats = executor.run_episode(max_steps=10000)
print(f"Episode completed: {stats}")
```

### Sanity Check

Test the system before deployment:

```bash
# Test with random policy
python U10_sanity_check_decentralized.py

# Test with trained policy
python U10_sanity_check_decentralized.py \
    --model-path ./models/u10/ppo_u10_final.zip \
    --vecnormalize-path ./models/u10/vecnormalize_u10_final.pkl

# Quick test (100 steps)
python U10_sanity_check_decentralized.py --max-steps 100

# Verbose mode (see all decisions)
python U10_sanity_check_decentralized.py --verbose
```

## Observation Space

### Local Observation (CTDE Mode)

Each drone receives a local observation with 3 components:

1. **drone_state** (8 features):
   - Position (x, y)
   - Battery level
   - Current load
   - Status (idle, flying, delivering, etc.)
   - Target location
   - Velocity

2. **candidates** (K × 12 features):
   - K candidate orders from MOPSO
   - Each order has 12 features:
     - Merchant location (x, y)
     - Customer location (x, y)
     - Deadline urgency
     - Priority
     - Distance
     - Estimated time
     - Profit
     - Load requirement
     - Weather impact
     - Merchant readiness

3. **global_context** (10 features):
   - Time of day (hour, minute, day_in_week, etc.)
   - Day progress (0-1)
   - Resource saturation (0-1)
   - Weather conditions (3 features)

**Key Property**: Observation size is **independent of number of drones N**

### Full Observation (Legacy Mode)

Includes all drones, orders, merchants, bases, plus:
- `current_drone_id`: Which drone is making decision

## Action Space

**Discrete(5)**: Single rule_id per decision

- **0**: Nearest order first
- **1**: Earliest deadline first
- **2**: Highest priority first
- **3**: Best profit/distance ratio
- **4**: Return to base / No order selection

**Key Property**: Action space **does not scale with N drones**

## Reward Structure

Multi-objective reward with 3 components:

1. **Objective 0 (Delivery Performance)**:
   - On-time deliveries: +1.0
   - Late deliveries: -0.5
   - Cancelled orders: -1.0

2. **Objective 1 (Energy Efficiency)**:
   - Energy saved: +reward
   - Energy wasted: -penalty
   - Battery return events: -0.5

3. **Objective 2 (Economic Profit)**:
   - Order revenue: +value
   - Flight cost: -distance × cost_per_unit
   - Cancellation penalty: -penalty

**Scalar Reward**: Weighted sum using fixed or conditioned weights
- Default fixed weights: (0.5, 0.3, 0.2)
- Conditioned mode: Random weights per episode for robustness

## Key Differences: CTDE vs Central Queue

| Aspect | Event-Driven Shared Policy (CTDE) | Central Queue (Legacy) |
|--------|-----------------------------------|------------------------|
| **Execution** | Decentralized - each drone independent | Centralized queue |
| **Action Space** | Discrete(5) - constant | Discrete(5) - constant |
| **Observation** | Local (drone, candidates, context) | Full + current_drone_id |
| **Scalability** | Scales to any N drones | Scales to any N drones |
| **Training Data** | Random sampling from decision drones | Sequential queue processing |
| **Deployment** | No central controller needed | Requires queue management |
| **Policy Sharing** | Explicit parameter sharing | Implicit via single policy |
| **Generalization** | Better across different N | Limited by training N |

## Troubleshooting

### Low Success Rate

If decision success rate is very low (<10%):
- Check candidate quality (MOPSO settings)
- Verify order generation rate
- Increase drone capacity
- Reduce number of drones relative to orders

### No Decisions Made

If no decisions occur during episode:
- Check decision point detection logic
- Verify drones reach idle state
- Ensure orders are being generated
- Check battery levels (low battery prevents decisions)

### Training Not Improving

If training reward doesn't improve:
- Increase training steps
- Adjust learning rate
- Check reward scaling (VecNormalize)
- Verify candidate diversity
- Try different drone sampling strategy

## References

- **PPO**: Schulman et al., "Proximal Policy Optimization Algorithms", 2017
- **MOPSO**: Coello & Lechuga, "MOPSO: A proposal for multiple objective particle swarm optimization", 2002
- **CTDE**: Lowe et al., "Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments", 2017

## Files

### Core Components
- `UAV_ENVIRONMENT_10.py`: Main environment with centralized arbitration
- `U10_candidate_generator.py`: MOPSO candidate generation (upper layer)
- `U10_single_uav_training_wrapper.py`: Training wrapper for CTDE
- `U10_decentralized_execution.py`: Decentralized execution manager
- `U10_train.py`: Training script with mode selection

### Legacy Components
- `U10_event_driven_single_uav_wrapper.py`: Legacy central queue wrapper

### Utilities
- `U10_sanity_check_decentralized.py`: Testing and validation script
- `U10_README.md`: This documentation file
