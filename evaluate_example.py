"""
Example script for evaluating a trained PPO model with VecNormalize.

This demonstrates how to properly load both the model and VecNormalize statistics
for evaluation or continued training.

IMPORTANT: You must load the VecNormalize statistics that were saved during training
to ensure consistent normalization. Without this, the model will receive improperly
scaled observations and rewards, leading to poor performance.

Usage:
    python evaluate_example.py --model-path ./models/u9_task/ppo_u9_task_final.zip \
                                --vecnormalize-path ./models/u9_task/vecnormalize_final.pkl \
                                --num-episodes 10
"""
import argparse
import os
import sys
import numpy as np

# Add repo root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize
from U9_train import make_env


def evaluate_model(model_path: str, vecnormalize_path: str, num_episodes: int = 10):
    """
    Evaluate a trained PPO model with proper VecNormalize loading.
    
    Args:
        model_path: Path to the saved model (.zip file)
        vecnormalize_path: Path to the saved VecNormalize statistics (.pkl file)
        num_episodes: Number of episodes to run for evaluation
    """
    print("=" * 70)
    print("Evaluation with VecNormalize")
    print("=" * 70)
    
    # Check if files exist
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not os.path.exists(vecnormalize_path):
        raise FileNotFoundError(f"VecNormalize stats not found: {vecnormalize_path}")
    
    print(f"Model path: {model_path}")
    print(f"VecNormalize path: {vecnormalize_path}")
    print(f"Number of episodes: {num_episodes}")
    print()
    
    # Create environment with same configuration as training
    # NOTE: You should use the same parameters as during training
    def env_fn():
        return make_env(
            seed=42,
            num_drones=20,  # Match training config
            obs_max_orders=400,
            top_k_merchants=100,
            candidate_k=20,
            rule_count=5,
            enable_random_events=False,
            debug_state_warnings=False,
            mopso_max_orders=400,
            mopso_max_orders_per_drone=5,
            fallback_policy="cargo_first",
            debug_stats_interval=0,
            enable_legacy_fallback=False,
            enable_diagnostics=False,
            diagnostics_interval=100,
            energy_e0=0.1,
            energy_alpha=0.5,
            battery_return_threshold=10.0,
        )
    
    # Create vectorized environment
    env = DummyVecEnv([env_fn])
    env = VecMonitor(env)
    
    # CRITICAL: Load VecNormalize statistics
    # This ensures the environment applies the same normalization as during training
    print("Loading VecNormalize statistics...")
    env = VecNormalize.load(vecnormalize_path, env)
    
    # For evaluation, disable training mode and reward normalization
    env.training = False
    env.norm_reward = False
    print("✓ VecNormalize loaded (training=False, norm_reward=False)")
    print()
    
    # Load the trained model
    print("Loading trained model...")
    model = PPO.load(model_path, env=env)
    print("✓ Model loaded")
    print()
    
    # Evaluate the model
    print(f"Running evaluation for {num_episodes} episodes...")
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        
        while not done:
            # Use deterministic actions for evaluation
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward[0]
            episode_length += 1
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        print(f"Episode {episode + 1}/{num_episodes}: reward={episode_reward:.2f}, length={episode_length}")
    
    env.close()
    
    # Print summary statistics
    print()
    print("=" * 70)
    print("Evaluation Summary")
    print("=" * 70)
    print(f"Mean episode reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Mean episode length: {np.mean(episode_lengths):.2f} ± {np.std(episode_lengths):.2f}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained PPO model with VecNormalize"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the saved model (.zip file)",
    )
    parser.add_argument(
        "--vecnormalize-path",
        type=str,
        required=True,
        help="Path to the saved VecNormalize statistics (.pkl file)",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=10,
        help="Number of episodes to run for evaluation (default: 10)",
    )
    
    args = parser.parse_args()
    evaluate_model(args.model_path, args.vecnormalize_path, args.num_episodes)


if __name__ == "__main__":
    main()
