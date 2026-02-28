"""
Decentralized Event-Driven Execution for U10

This module implements the event-driven decentralized execution architecture where:
1. Each drone independently makes decisions using a shared policy
2. Decisions are made only when drones reach decision points (event-driven)
3. System fast-forwards time when no decisions are needed
4. Centralized arbitration ensures atomic order assignment (READY -> ASSIGNED)

Key Concepts:
- CTDE (Centralized Training, Decentralized Execution): Train one policy, execute independently per drone
- Event-driven: Only act when needed, not every time step
- Shared policy: All drones use the same policy weights but with their own local observations
- Centralized arbitration: Environment atomically handles order conflicts

Usage:
    # For evaluation/deployment with a trained policy
    executor = DecentralizedEventDrivenExecutor(env, policy)
    obs, info = executor.reset()
    
    while not done:
        obs, reward, terminated, truncated, info = executor.step()
        done = terminated or truncated
"""

from typing import Dict, List, Optional, Any, Tuple, Callable
import numpy as np
import gymnasium as gym


class DecentralizedEventDrivenExecutor:
    """
    Event-driven decentralized execution manager for multi-UAV system.
    
    This executor:
    1. Detects drones at decision points using env.get_decision_drones()
    2. For each decision drone, extracts local observation and calls policy
    3. Submits rule_id to centralized arbitration via env.apply_rule_to_drone()
    4. Fast-forwards environment when no drones need decisions
    
    Args:
        env: The UAV environment (must have get_decision_drones, apply_rule_to_drone)
        policy_fn: Function that takes local_obs and returns rule_id (0-4)
        max_skip_steps: Maximum steps to skip when waiting for decisions
        verbose: Whether to print execution details
    """
    
    def __init__(
        self,
        env: gym.Env,
        policy_fn: Callable[[Dict], int],
        max_skip_steps: int = 10,
        verbose: bool = False
    ):
        self.env = env
        self.policy_fn = policy_fn
        self.max_skip_steps = max_skip_steps
        self.verbose = verbose
        
        # Unwrap environment to access methods
        self.unwrapped_env = env.unwrapped
        
        # Statistics
        self.total_decisions = 0
        self.total_decision_rounds = 0
        self.total_skip_steps = 0
        self.successful_decisions = 0
        self.failed_decisions = 0
        self.decision_failures_by_reason = {}
        
        # Episode state
        self.episode_active = False
        self.cumulative_reward = 0.0
        
    def reset(self, **kwargs) -> Tuple[Dict, Dict]:
        """
        Reset environment and prepare for episode.
        
        Returns:
            observation: Full environment observation (for monitoring)
            info: Info dict with execution metadata
        """
        obs, info = self.env.reset(**kwargs)
        
        # Reset statistics
        self.total_decisions = 0
        self.total_decision_rounds = 0
        self.total_skip_steps = 0
        self.successful_decisions = 0
        self.failed_decisions = 0
        self.decision_failures_by_reason = {}
        self.cumulative_reward = 0.0
        self.episode_active = True
        
        # Add executor info
        info['executor'] = self.get_statistics()
        
        if self.verbose:
            print("=" * 60)
            print("DecentralizedEventDrivenExecutor: Episode Started")
            print("=" * 60)
        
        return obs, info
    
    def step(self) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        Execute one decision round for all drones at decision points.
        
        This method:
        1. Gets all drones at decision points
        2. For each drone: extract local obs, call policy, submit to arbitrator
        3. Advances environment one step after all decisions
        4. If no decisions, fast-forwards until next decision event
        
        Returns:
            observation: Full environment observation
            reward: Accumulated reward from this step
            terminated: Whether episode terminated
            truncated: Whether episode truncated
            info: Info dict with decision details
        """
        if not self.episode_active:
            raise RuntimeError("Episode not active. Call reset() first.")
        
        # Get drones at decision points
        decision_drones = self.unwrapped_env.get_decision_drones()
        
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}
        
        if decision_drones:
            # Process all drones at decision points
            obs, reward, terminated, truncated, info = self._process_decision_round(decision_drones)
            total_reward += reward
        else:
            # No decisions needed, fast-forward to next decision event
            obs, reward, terminated, truncated, info = self._skip_to_next_decision()
            total_reward += reward
        
        # Update cumulative reward
        self.cumulative_reward += total_reward
        
        # Add executor statistics to info
        info['executor'] = self.get_statistics()
        
        # Check if episode ended
        if terminated or truncated:
            self.episode_active = False
            if self.verbose:
                print("=" * 60)
                print("DecentralizedEventDrivenExecutor: Episode Ended")
                print(f"  Total Decisions: {self.total_decisions}")
                print(f"  Successful: {self.successful_decisions}, Failed: {self.failed_decisions}")
                print(f"  Decision Rounds: {self.total_decision_rounds}")
                print(f"  Skip Steps: {self.total_skip_steps}")
                print(f"  Cumulative Reward: {self.cumulative_reward:.2f}")
                print("=" * 60)
        
        return obs, total_reward, terminated, truncated, info
    
    def _process_decision_round(
        self, 
        decision_drones: List[int]
    ) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        Process one decision round for all drones at decision points.
        
        Args:
            decision_drones: List of drone IDs at decision points
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        self.total_decision_rounds += 1
        round_decisions = []
        
        if self.verbose:
            print(f"\n--- Decision Round {self.total_decision_rounds} ---")
            print(f"  Drones at decision points: {decision_drones}")
        
        # Get current observation for extracting local obs
        current_obs = self._get_current_observation()
        
        # Process each drone independently
        for drone_id in decision_drones:
            self.total_decisions += 1
            
            # Extract local observation for this drone
            local_obs = self._extract_local_observation(current_obs, drone_id)
            
            # Call policy to get rule_id
            rule_id = self.policy_fn(local_obs)
            
            # Submit to centralized arbitrator
            success = self.unwrapped_env.apply_rule_to_drone(drone_id, rule_id)
            
            # Track result
            if success:
                self.successful_decisions += 1
                decision_result = "SUCCESS"
            else:
                self.failed_decisions += 1
                decision_result = "FAILED"
                # Track failure reason (if available in future)
                reason = "unknown"
                self.decision_failures_by_reason[reason] = \
                    self.decision_failures_by_reason.get(reason, 0) + 1
            
            round_decisions.append({
                'drone_id': drone_id,
                'rule_id': rule_id,
                'success': success
            })
            
            if self.verbose:
                print(f"  Drone {drone_id}: rule_id={rule_id}, result={decision_result}")
        
        # Advance environment one step after all decisions
        # Use dummy action (all zeros) to just advance time
        dummy_action = np.zeros(self.unwrapped_env.num_drones, dtype=np.int32)
        obs, reward, terminated, truncated, info = self.env.step(dummy_action)
        
        # Add decision round info
        info['decision_round'] = {
            'round_number': self.total_decision_rounds,
            'num_decisions': len(decision_drones),
            'decisions': round_decisions
        }
        
        return obs, reward, terminated, truncated, info
    
    def _skip_to_next_decision(self) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        Fast-forward environment until a decision event occurs.
        
        Returns:
            observation, accumulated_reward, terminated, truncated, info
        """
        total_reward = 0.0
        terminated = False
        truncated = False
        obs = None
        info = {}
        
        for skip_step in range(self.max_skip_steps):
            # Advance environment with no-op
            dummy_action = np.zeros(self.unwrapped_env.num_drones, dtype=np.int32)
            obs, reward, terminated, truncated, info = self.env.step(dummy_action)
            
            total_reward += reward
            self.total_skip_steps += 1
            
            # Check if episode ended
            if terminated or truncated:
                break
            
            # Check for decision events
            decision_drones = self.unwrapped_env.get_decision_drones()
            if decision_drones:
                # Found decision event, will be processed in next step() call
                break
        
        if self.verbose and self.total_skip_steps > 0:
            print(f"  Skipped {skip_step + 1} steps (no decisions)")
        
        info['skip_info'] = {
            'steps_skipped': skip_step + 1 if not (terminated or truncated) else skip_step,
            'reason': 'episode_end' if (terminated or truncated) else 'decision_event_found'
        }
        
        return obs, total_reward, terminated, truncated, info
    
    def _get_current_observation(self) -> Dict:
        """
        Get current observation from environment.
        
        Returns:
            Current observation dict
        """
        # Access the current observation from environment
        # This assumes the environment stores its last observation
        if hasattr(self.unwrapped_env, 'last_obs'):
            return self.unwrapped_env.last_obs
        elif hasattr(self.unwrapped_env, '_get_obs'):
            return self.unwrapped_env._get_obs()
        else:
            # Fallback: call observation method directly
            return self.unwrapped_env._get_observation()
    
    def _extract_local_observation(self, full_obs: Dict, drone_id: int) -> Dict:
        """
        Extract local observation for a specific drone.
        
        This creates a homogeneous observation that includes:
        - Drone's own state (8 features)
        - Drone's candidate orders (K x 12 features)
        - Global context (time, weather, resource saturation)
        
        Args:
            full_obs: Full observation from environment
            drone_id: Drone ID to extract observation for
            
        Returns:
            Local observation dict with keys: drone_state, candidates, global_context
        """
        # Extract drone's own state
        drone_state = full_obs['drones'][drone_id]
        
        # Extract drone's candidates
        candidates = full_obs['candidates'][drone_id]
        
        # Build global context (summary statistics)
        global_context = np.concatenate([
            full_obs['time'],  # 5 dims: hour, minute, day, day_in_week, month
            full_obs['day_progress'],  # 1 dim
            full_obs['resource_saturation'],  # 1 dim
            full_obs['weather_details'][:3],  # 3 dims (first 3 weather features)
        ])
        
        local_obs = {
            'drone_state': drone_state.astype(np.float32),
            'candidates': candidates.astype(np.float32),
            'global_context': global_context.astype(np.float32),
        }
        
        return local_obs
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get execution statistics.
        
        Returns:
            Dictionary with statistics
        """
        return {
            'total_decisions': self.total_decisions,
            'total_decision_rounds': self.total_decision_rounds,
            'total_skip_steps': self.total_skip_steps,
            'successful_decisions': self.successful_decisions,
            'failed_decisions': self.failed_decisions,
            'success_rate': (
                self.successful_decisions / max(self.total_decisions, 1)
            ),
            'failure_reasons': dict(self.decision_failures_by_reason),
            'cumulative_reward': self.cumulative_reward,
        }
    
    def run_episode(self, max_steps: int = 10000) -> Dict[str, Any]:
        """
        Run a complete episode from start to finish.
        
        Args:
            max_steps: Maximum number of decision steps
            
        Returns:
            Episode statistics
        """
        obs, info = self.reset()
        
        for step_num in range(max_steps):
            obs, reward, terminated, truncated, info = self.step()
            
            if terminated or truncated:
                break
        
        return self.get_statistics()
