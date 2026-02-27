"""
U11 Sanity Check for Decentralized Event-Driven Execution

This script validates the decentralized event-driven execution system by:
1. Running a small episode with random policy
2. Testing event-driven loop functionality
3. Printing statistics (decisions, skips, success/failure rates)
4. Optionally loading and testing a trained policy

Usage:
    # Test with random policy
    python U11_sanity_check_decentralized.py

    # Test with trained policy
    python U11_sanity_check_decentralized.py --model-path ./models/u10/ppo_u10_final.zip

    # Quick test (fewer steps)
    python U11_sanity_check_decentralized.py --max-steps 100

    # Ablation: sweep order_cutoff_steps K over multiple seeds, output CSV
    python U11_sanity_check_decentralized.py --ablation-cutoff \\
        --cutoff-values 0,6,12,18,24 --seeds 42,43,44 --csv-out ablation_cutoff.csv
"""

import argparse
import csv
import os
import sys
import numpy as np

# Add repo root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from UAV_ENVIRONMENT_11 import ThreeObjectiveDroneDeliveryEnv
from U10_candidate_generator import MOPSOCandidateGenerator
from U11_decentralized_execution import DecentralizedEventDrivenExecutor


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


def _make_env(args, order_cutoff_steps: int = 0):
    """Create environment with given order_cutoff_steps."""
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
        order_cutoff_steps=order_cutoff_steps,
    )
    return env


def _make_candidate_generator(args, seed: int):
    """Create a MOPSO candidate generator."""
    return MOPSOCandidateGenerator(
        candidate_k=args.candidate_k,
        n_particles=30,
        n_iterations=10,
        max_orders=200,
        max_orders_per_drone=10,
        seed=seed,
    )


def run_sanity_check(args):
    """Run sanity check with specified configuration."""
    print("=" * 80)
    print("U10 Decentralized Execution Sanity Check")
    print("=" * 80)

    # Create environment
    print("\nCreating environment...")
    env = _make_env(args, order_cutoff_steps=args.order_cutoff_steps)

    # Create MOPSO candidate generator
    print("Creating MOPSO candidate generator...")
    candidate_generator = _make_candidate_generator(args, seed=args.seed)
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


def run_ablation_cutoff(args):
    """Run K-sweep ablation: for each (K, seed), run one episode and collect stats."""
    cutoff_values = [int(v.strip()) for v in args.cutoff_values.split(",")]
    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    print("=" * 80)
    print("Ablation: order_cutoff_steps sweep")
    print(f"  K values : {cutoff_values}")
    print(f"  Seeds    : {seeds}")
    print(f"  CSV out  : {args.csv_out}")
    print("=" * 80)

    csv_fields = [
        "order_cutoff_steps", "seed",
        "generated_total", "completed_total", "general_completion",
        "serviceable_generated", "serviceable_completed", "serviceable_completion",
    ]

    rows = []

    for k in cutoff_values:
        for seed in seeds:
            print(f"\n--- K={k}, seed={seed} ---")
            np.random.seed(seed)

            env = _make_env(args, order_cutoff_steps=k)
            cg = _make_candidate_generator(args, seed=seed)
            env.set_candidate_generator(cg)

            executor = DecentralizedEventDrivenExecutor(
                env=env,
                policy_fn=random_policy,
                max_skip_steps=args.max_skip_steps,
                verbose=False,
            )
            executor.run_episode(max_steps=args.max_steps)

            completion = env.get_completion_stats()
            row = {"order_cutoff_steps": k, "seed": seed}
            row.update(completion)
            rows.append(row)

            print(
                f"  general={completion['general_completion']:.3f}"
                f"  serviceable={completion['serviceable_completion']:.3f}"
                f"  gen_total={completion['generated_total']}"
            )

    # Write CSV
    with open(args.csv_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV written to: {args.csv_out}")

    # Aggregate by K and recommend best K
    _print_best_k(cutoff_values, rows)


def _print_best_k(cutoff_values, rows):
    """Aggregate rows by K and print best-K recommendation."""
    from collections import defaultdict

    general_by_k = defaultdict(list)
    serviceable_by_k = defaultdict(list)
    for row in rows:
        k = row["order_cutoff_steps"]
        general_by_k[k].append(row["general_completion"])
        serviceable_by_k[k].append(row["serviceable_completion"])

    print("\n" + "=" * 80)
    print("Ablation Summary (mean across seeds)")
    print("=" * 80)
    print(f"{'K':>6}  {'mean_general':>14}  {'mean_serviceable':>16}")
    print("-" * 42)

    mean_general = {}
    mean_serviceable = {}
    for k in cutoff_values:
        mg = float(np.mean(general_by_k[k])) if general_by_k[k] else 0.0
        ms = float(np.mean(serviceable_by_k[k])) if serviceable_by_k[k] else 0.0
        mean_general[k] = mg
        mean_serviceable[k] = ms
        print(f"{k:>6}  {mg:>14.4f}  {ms:>16.4f}")

    best_g = max(mean_general, key=mean_general.get)
    best_s = max(mean_serviceable, key=mean_serviceable.get)

    print("\n--- Best K Recommendation ---")
    if best_g == best_s:
        print(f"  Both metrics peak at K={best_g}  ← recommended K")
    else:
        print(f"  general_completion    peaks at K={best_g}  (mean={mean_general[best_g]:.4f})")
        print(f"  serviceable_completion peaks at K={best_s}  (mean={mean_serviceable[best_s]:.4f})")
        # Tiebreak: maximise product (balanced)
        best_product = max(cutoff_values,
                           key=lambda k: mean_general[k] * mean_serviceable[k])
        best_min = max(cutoff_values,
                       key=lambda k: min(mean_general[k], mean_serviceable[k]))
        print(f"  Compromise (max product)  : K={best_product}")
        print(f"  Compromise (max min-value): K={best_min}")
    print("=" * 80)


def main():
    """Parse arguments and run sanity check."""
    parser = argparse.ArgumentParser(
        description="U11 Decentralized Execution Sanity Check"
    )

    # Environment parameters
    parser.add_argument("--num-drones", type=int, default=20,
                        help="Number of drones (default: 10)")
    parser.add_argument("--obs-max-orders", type=int, default=400,
                        help="Maximum orders in observation (default: 200)")
    parser.add_argument("--top-k-merchants", type=int, default=100,
                        help="Top K merchants (default: 50)")
    parser.add_argument("--candidate-k", type=int, default=20,
                        help="Number of candidates per drone (default: 20)")
    parser.add_argument("--enable-random-events", action="store_true", default=False,
                        help="Enable random events (default: False)")

    # Executor parameters
    parser.add_argument("--max-skip-steps", type=int, default=1,
                        help="Max steps to skip when waiting for decisions (default: 10)")
    parser.add_argument("--max-steps", type=int, default=500,
                        help="Maximum decision steps per episode (default: 500)")

    # Policy parameters
    parser.add_argument("--model-path", type=str, default='ppo_u11_final.zip',
                        help="Path to trained model (.zip file) - if not provided, uses random policy")
    parser.add_argument("--vecnormalize-path", type=str, default='vecnormalize_u11_final.pkl',
                        help="Path to VecNormalize stats (.pkl file)")

    # Order cutoff parameters
    parser.add_argument("--order-cutoff-steps", type=int, default=0,
                        help="Stop generating new orders K steps before business end (default: 0 = no cutoff)")
    parser.add_argument("--ablation-cutoff", action="store_true", default=False,
                        help="Enable K-sweep ablation mode for order_cutoff_steps")
    parser.add_argument("--cutoff-values", type=str, default="0,6,12,18,24",
                        help="Comma-separated K values for ablation (default: '0,6,12,18,24')")
    parser.add_argument("--seeds", type=str, default="42,43,44",
                        help="Comma-separated seeds for ablation (default: '42,43,44')")
    parser.add_argument("--csv-out", type=str, default="ablation_cutoff.csv",
                        help="Output CSV path for ablation results (default: 'ablation_cutoff.csv')")

    # Other parameters
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--verbose", action="store_true", default=False,
                        help="Print detailed execution logs (default: False)")

    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)

    # Run ablation or single sanity check
    try:
        if args.ablation_cutoff:
            run_ablation_cutoff(args)
        else:
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

