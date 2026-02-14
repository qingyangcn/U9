"""
Event-Driven Single UAV PPO Training Script

This script demonstrates how to train a PPO agent using the event-driven
single UAV wrapper with candidate-based filtering.

Features:
- Event-driven decision making (Discrete(5) action space)
- Candidate-based order filtering
- Homogeneous policy (same policy for all drones)
- Configurable candidate generation strategy

Usage:
    python scripts/train_event_driven_rule_ppo.py --total-steps 10000 --seed 42
    python scripts/train_event_driven_rule_ppo.py --candidate-strategy mixed --candidate-k 10
"""

import argparse
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import numpy as np
    import gymnasium as gym
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    print("Warning: stable-baselines3 not available. Running in demo mode only.")

from UAV_ENVIRONMENT_9 import ThreeObjectiveDroneDeliveryEnv
from candidate_generator import (
    NearestCandidateGenerator,
    EarliestDeadlineCandidateGenerator,
    MixedHeuristicCandidateGenerator,
    PSOMOPSOCandidateGenerator
)
from wrappers import EventDrivenSingleUAVWrapper


def create_env(
    num_drones=3,
    grid_size=10,
    candidate_strategy='mixed',
    candidate_k=20,
    candidate_fallback=True,
    max_skip_steps=10,
    local_observation=False,
    seed=None
):
    """
    Create event-driven UAV environment with candidate filtering.
    
    Args:
        num_drones: Number of drones
        grid_size: Grid size
        candidate_strategy: Candidate generation strategy 
                          ('nearest', 'earliest', 'mixed', 'pso')
        candidate_k: Number of candidates per drone
        candidate_fallback: Allow fallback to all orders if candidates empty
        max_skip_steps: Maximum steps to skip when waiting for decisions
        local_observation: Use local observation instead of global
        seed: Random seed
        
    Returns:
        Wrapped environment
    """
    # Create base environment
    base_env = ThreeObjectiveDroneDeliveryEnv(
        grid_size=grid_size,
        num_drones=num_drones,
        max_orders=100,
        steps_per_hour=4,
        enable_random_events=False,  # Disable for training stability
        candidate_fallback_enabled=candidate_fallback,
        candidate_update_interval=1,  # Update every step
        reward_output_mode="scalar",  # Required for SB3
    )
    
    # Create candidate generator based on strategy
    if candidate_strategy == 'nearest':
        generator = NearestCandidateGenerator(candidate_k=candidate_k)
    elif candidate_strategy == 'earliest':
        generator = EarliestDeadlineCandidateGenerator(candidate_k=candidate_k)
    elif candidate_strategy == 'mixed':
        generator = MixedHeuristicCandidateGenerator(
            candidate_k=candidate_k,
            distance_weight=0.5,
            deadline_weight=0.5
        )
    elif candidate_strategy == 'pso':
        generator = PSOMOPSOCandidateGenerator(candidate_k=candidate_k)
    else:
        raise ValueError(f"Unknown candidate strategy: {candidate_strategy}")
    
    # Set candidate generator
    base_env.set_candidate_generator(generator)
    
    # Wrap with event-driven wrapper
    env = EventDrivenSingleUAVWrapper(
        base_env,
        max_skip_steps=max_skip_steps,
        local_observation=local_observation
    )
    
    return env


def run_demo(args):
    """Run a simple demonstration without training."""
    print("=" * 60)
    print("Event-Driven UAV Environment Demo")
    print("=" * 60)
    
    # Create environment
    env = create_env(
        num_drones=args.num_drones,
        grid_size=args.grid_size,
        candidate_strategy=args.candidate_strategy,
        candidate_k=args.candidate_k,
        candidate_fallback=args.candidate_fallback,
        max_skip_steps=args.max_skip_steps,
        local_observation=args.local_observation,
        seed=args.seed
    )
    
    print(f"\nEnvironment created:")
    print(f"  - Drones: {args.num_drones}")
    print(f"  - Grid size: {args.grid_size}")
    print(f"  - Candidate strategy: {args.candidate_strategy}")
    print(f"  - Candidate K: {args.candidate_k}")
    print(f"  - Action space: {env.action_space}")
    print(f"  - Max skip steps: {args.max_skip_steps}")
    print(f"  - Local observation: {args.local_observation}")
    
    # Reset environment
    obs, info = env.reset(seed=args.seed)
    print(f"\nInitial state:")
    print(f"  - Decision queue length: {info.get('decision_queue_length', 0)}")
    print(f"  - Current drone: {info.get('current_drone_id', -1)}")
    
    # Run random policy for a few steps
    print(f"\nRunning random policy for {args.demo_steps} steps...")
    total_reward = 0.0
    for step in range(args.demo_steps):
        # Random action
        action = env.action_space.sample()
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if step % 10 == 0:
            print(f"  Step {step}: reward={reward:.4f}, "
                  f"queue={info.get('decision_queue_length', 0)}, "
                  f"drone={info.get('current_drone_id', -1)}")
        
        if terminated or truncated:
            print(f"  Episode ended at step {step}")
            break
    
    # Print statistics
    stats = env.get_statistics()
    print(f"\nFinal statistics:")
    print(f"  - Total reward: {total_reward:.4f}")
    print(f"  - Total decisions: {stats['total_decisions']}")
    print(f"  - Total skips: {stats['total_skips']}")
    print(f"  - Queue length: {stats['queue_length']}")
    
    print("\nDemo completed successfully!")


def run_training(args):
    """Run PPO training."""
    if not SB3_AVAILABLE:
        print("Error: stable-baselines3 is required for training.")
        print("Install with: pip install stable-baselines3")
        return
    
    print("=" * 60)
    print("Event-Driven UAV PPO Training")
    print("=" * 60)
    
    # Create vectorized environment
    def make_env():
        return create_env(
            num_drones=args.num_drones,
            grid_size=args.grid_size,
            candidate_strategy=args.candidate_strategy,
            candidate_k=args.candidate_k,
            candidate_fallback=args.candidate_fallback,
            max_skip_steps=args.max_skip_steps,
            local_observation=args.local_observation,
            seed=args.seed
        )
    
    env = DummyVecEnv([make_env])
    
    print(f"\nTraining configuration:")
    print(f"  - Total timesteps: {args.total_steps}")
    print(f"  - Drones: {args.num_drones}")
    print(f"  - Candidate strategy: {args.candidate_strategy}")
    print(f"  - Candidate K: {args.candidate_k}")
    
    # Create PPO model
    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        tensorboard_log=args.tensorboard_log
    )
    
    print("\nStarting training...")
    model.learn(total_timesteps=args.total_steps)
    
    # Save model
    if args.save_path:
        model.save(args.save_path)
        print(f"\nModel saved to: {args.save_path}")
    
    print("\nTraining completed successfully!")


def main():
    parser = argparse.ArgumentParser(
        description="Train PPO agent with event-driven single UAV wrapper"
    )
    
    # Environment parameters
    parser.add_argument('--num-drones', type=int, default=3,
                       help='Number of drones (default: 3)')
    parser.add_argument('--grid-size', type=int, default=10,
                       help='Grid size (default: 10)')
    
    # Candidate generation parameters
    parser.add_argument('--candidate-strategy', type=str, default='mixed',
                       choices=['nearest', 'earliest', 'mixed', 'pso'],
                       help='Candidate generation strategy (default: mixed)')
    parser.add_argument('--candidate-k', type=int, default=20,
                       help='Number of candidates per drone (default: 20)')
    parser.add_argument('--no-candidate-fallback', action='store_true',
                       help='Disable fallback to all orders when candidates empty')
    
    # Wrapper parameters
    parser.add_argument('--max-skip-steps', type=int, default=10,
                       help='Max steps to skip waiting for decisions (default: 10)')
    parser.add_argument('--local-observation', action='store_true',
                       help='Use local observation instead of global')
    
    # Training parameters
    parser.add_argument('--mode', type=str, default='demo',
                       choices=['demo', 'train'],
                       help='Run mode: demo or train (default: demo)')
    parser.add_argument('--total-steps', type=int, default=10000,
                       help='Total training timesteps (default: 10000)')
    parser.add_argument('--learning-rate', type=float, default=3e-4,
                       help='Learning rate (default: 3e-4)')
    parser.add_argument('--n-steps', type=int, default=2048,
                       help='PPO n_steps (default: 2048)')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='PPO batch size (default: 64)')
    parser.add_argument('--n-epochs', type=int, default=10,
                       help='PPO n_epochs (default: 10)')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='Discount factor (default: 0.99)')
    parser.add_argument('--tensorboard-log', type=str, default=None,
                       help='Tensorboard log directory')
    parser.add_argument('--save-path', type=str, default='event_driven_ppo',
                       help='Model save path (default: event_driven_ppo)')
    
    # Demo parameters
    parser.add_argument('--demo-steps', type=int, default=100,
                       help='Number of steps for demo mode (default: 100)')
    
    # General parameters
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    
    args = parser.parse_args()
    args.candidate_fallback = not args.no_candidate_fallback
    
    # Run appropriate mode
    if args.mode == 'demo':
        run_demo(args)
    else:
        run_training(args)


if __name__ == '__main__':
    main()
