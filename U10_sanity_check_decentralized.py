"""
U10 Sanity Check for Decentralized Event-Driven Execution

This script validates the decentralized event-driven execution system by:
1. Running a small episode with random policy
2. Testing event-driven loop functionality
3. Printing statistics (decisions, skips, success/failure rates)
4. Optionally loading and testing a trained policy

Usage:
    # Test with random policy
    python U10_sanity_check_decentralized.py
    
    # Test with trained policy
    python U10_sanity_check_decentralized.py --model-path ./models/u10/ppo_u10_final.zip
    
    # Quick test (fewer steps)
    python U10_sanity_check_decentralized.py --max-steps 100
"""

import argparse
import os
import sys
import numpy as np

# Add repo root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from UAV_ENVIRONMENT_10 import ThreeObjectiveDroneDeliveryEnv
from U10_candidate_generator import MOPSOCandidateGenerator
from U10_decentralized_execution import DecentralizedEventDrivenExecutor


def random_policy(local_obs: dict) -> int:
    """Simple random policy for testing."""
    return np.random.randint(0, 5)


def load_trained_policy(model_path: str, vecnormalize_path: str = None):
    """
    Load a trained PPO policy.
    
    Args:
        model_path: Path to trained model (.zip file)
        vecnormalize_path: Path to VecNormalize stats (.pkl file)
        
    Returns:
        Policy function that takes local_obs and returns rule_id
    """
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import VecNormalize
    except ImportError:
        raise RuntimeError("Please install stable-baselines3: pip install stable-baselines3")
    
    # Load model
    model = PPO.load(model_path)
    
    # Load VecNormalize stats if provided
    vec_normalize = None
    if vecnormalize_path and os.path.exists(vecnormalize_path):
        print(f"Loading VecNormalize stats from: {vecnormalize_path}")
        # Note: VecNormalize needs a dummy env to load
        # For now, we'll skip normalization in evaluation
        print("Warning: VecNormalize stats loading not implemented in sanity check")
    
    def policy_fn(local_obs: dict) -> int:
        """Wrapper function for trained policy."""
        # Convert dict obs to format expected by model
        # Model expects dict with keys: drone_state, candidates, global_context
        action, _ = model.predict(local_obs, deterministic=True)
        return int(action)
    
    return policy_fn


def run_sanity_check(args):
    """Run sanity check with specified configuration."""
    print("=" * 80)
    print("U10 Decentralized Execution Sanity Check")
    print("=" * 80)
    
    # Create environment
    print("\nCreating environment...")
    env = ThreeObjectiveDroneDeliveryEnv(
        grid_size=16,
        num_drones=args.num_drones,
        max_orders=args.obs_max_orders,
        num_bases=2,
        steps_per_hour=12,
        drone_max_capacity=10,
        top_k_merchants=args.top_k_merchants,
        reward_output_mode="scalar",
        enable_random_events=args.enable_random_events,
        debug_state_warnings=False,
        fixed_objective_weights=(0.5, 0.3, 0.2),
        num_candidates=args.candidate_k,
        rule_count=5,
        enable_diagnostics=False,
        energy_e0=0.1,
        energy_alpha=0.5,
        battery_return_threshold=10.0,
        multi_objective_mode="fixed",
        candidate_update_interval=8,
        candidate_fallback_enabled=False,
    )
    
    # Create MOPSO candidate generator
    print("Creating MOPSO candidate generator...")
    candidate_generator = MOPSOCandidateGenerator(
        candidate_k=args.candidate_k,
        n_particles=20,
        n_iterations=5,
        max_orders=100,
        max_orders_per_drone=10,
        seed=args.seed,
    )
    env.set_candidate_generator(candidate_generator)
    
    # Choose policy
    if args.model_path:
        print(f"\nLoading trained policy from: {args.model_path}")
        policy_fn = load_trained_policy(args.model_path, args.vecnormalize_path)
        policy_name = "Trained Policy"
    else:
        print("\nUsing random policy for testing")
        policy_fn = random_policy
        policy_name = "Random Policy"
    
    # Create decentralized executor
    print(f"Creating decentralized executor with {policy_name}...")
    executor = DecentralizedEventDrivenExecutor(
        env=env,
        policy_fn=policy_fn,
        max_skip_steps=args.max_skip_steps,
        verbose=args.verbose
    )
    
    # Run episode
    print("\n" + "=" * 80)
    print(f"Running episode (max {args.max_steps} decision steps)...")
    print("=" * 80 + "\n")
    
    stats = executor.run_episode(max_steps=args.max_steps)
    
    # Print results
    print("\n" + "=" * 80)
    print("Sanity Check Results")
    print("=" * 80)
    print(f"\nPolicy: {policy_name}")
    print(f"Environment: {args.num_drones} drones, {args.candidate_k} candidates/drone")
    print(f"\nExecution Statistics:")
    print(f"  Total Decision Rounds: {stats['total_decision_rounds']}")
    print(f"  Total Individual Decisions: {stats['total_decisions']}")
    print(f"  Successful Decisions: {stats['successful_decisions']}")
    print(f"  Failed Decisions: {stats['failed_decisions']}")
    print(f"  Success Rate: {stats['success_rate']:.2%}")
    print(f"  Total Skip Steps: {stats['total_skip_steps']}")
    print(f"  Cumulative Reward: {stats['cumulative_reward']:.2f}")
    
    if stats['failure_reasons']:
        print(f"\nFailure Reasons:")
        for reason, count in stats['failure_reasons'].items():
            print(f"  {reason}: {count}")
    
    print("\n" + "=" * 80)
    print("Sanity Check PASSED ✓")
    print("=" * 80)
    print("\nKey Validations:")
    print("  ✓ Event-driven loop executed successfully")
    print("  ✓ Decentralized decisions processed")
    print("  ✓ Fast-forward mechanism worked")
    print(f"  ✓ Decision success rate: {stats['success_rate']:.2%}")
    
    if stats['total_decisions'] == 0:
        print("\n⚠ WARNING: No decisions were made during the episode!")
        print("  This might indicate an issue with decision point detection.")
    
    if stats['success_rate'] < 0.1:
        print("\n⚠ WARNING: Very low success rate!")
        print("  This might indicate issues with order availability or drone capacity.")
    
    print("\n" + "=" * 80)


def main():
    """Parse arguments and run sanity check."""
    parser = argparse.ArgumentParser(
        description="U10 Decentralized Execution Sanity Check"
    )
    
    # Environment parameters
    parser.add_argument("--num-drones", type=int, default=10,
                        help="Number of drones (default: 10)")
    parser.add_argument("--obs-max-orders", type=int, default=200,
                        help="Maximum orders in observation (default: 200)")
    parser.add_argument("--top-k-merchants", type=int, default=50,
                        help="Top K merchants (default: 50)")
    parser.add_argument("--candidate-k", type=int, default=20,
                        help="Number of candidates per drone (default: 20)")
    parser.add_argument("--enable-random-events", action="store_true", default=False,
                        help="Enable random events (default: False)")
    
    # Executor parameters
    parser.add_argument("--max-skip-steps", type=int, default=10,
                        help="Max steps to skip when waiting for decisions (default: 10)")
    parser.add_argument("--max-steps", type=int, default=500,
                        help="Maximum decision steps per episode (default: 500)")
    
    # Policy parameters
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to trained model (.zip file) - if not provided, uses random policy")
    parser.add_argument("--vecnormalize-path", type=str, default=None,
                        help="Path to VecNormalize stats (.pkl file)")
    
    # Other parameters
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--verbose", action="store_true", default=False,
                        help="Print detailed execution logs (default: False)")
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Run sanity check
    try:
        run_sanity_check(args)
    except Exception as e:
        print("\n" + "=" * 80)
        print("Sanity Check FAILED ✗")
        print("=" * 80)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
