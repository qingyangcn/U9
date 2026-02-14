"""
Unit tests for EventDrivenSingleUAVWrapper.

Tests the event-driven wrapper functionality.
"""

import unittest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from UAV_ENVIRONMENT_9 import ThreeObjectiveDroneDeliveryEnv
from wrappers import EventDrivenSingleUAVWrapper


class TestEventDrivenWrapper(unittest.TestCase):
    """Test suite for EventDrivenSingleUAVWrapper."""
    
    def setUp(self):
        """Create environment and wrapper for testing."""
        base_env = ThreeObjectiveDroneDeliveryEnv(
            grid_size=10,
            num_drones=3,
            max_orders=50,
            steps_per_hour=4,
            enable_random_events=False,
        )
        self.env = EventDrivenSingleUAVWrapper(
            base_env,
            max_skip_steps=10,
            local_observation=False
        )
    
    def test_wrapper_initialization(self):
        """Test wrapper initializes correctly."""
        # Check action space is Discrete(5)
        from gymnasium import spaces
        self.assertIsInstance(self.env.action_space, spaces.Discrete)
        self.assertEqual(self.env.action_space.n, 5)
        
        # Check wrapper attributes
        self.assertEqual(self.env.max_skip_steps, 10)
        self.assertFalse(self.env.local_observation)
    
    def test_reset(self):
        """Test reset functionality."""
        obs, info = self.env.reset(seed=42)
        
        # Check observation is a dict
        self.assertIsInstance(obs, dict)
        
        # Check info contains metadata
        self.assertIn('decision_queue_length', info)
        self.assertIn('current_drone_id', info)
        
        # Check statistics reset
        self.assertEqual(self.env.total_decisions, 0)
        self.assertEqual(self.env.total_skips, 0)
    
    def test_step_with_decision(self):
        """Test step when drone is at decision point."""
        obs, info = self.env.reset(seed=42)
        
        # Take a step with rule 2 (READY_EDF)
        obs, reward, terminated, truncated, info = self.env.step(2)
        
        # Check observation is dict
        self.assertIsInstance(obs, dict)
        
        # Check reward is float
        self.assertIsInstance(reward, (float, np.floating))
        
        # Check terminated and truncated are bool
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)
        
        # Check info has metadata
        self.assertIn('decision_queue_length', info)
        self.assertIn('current_drone_id', info)
    
    def test_decision_queue(self):
        """Test decision queue mechanics."""
        obs, info = self.env.reset(seed=42)
        
        # Queue should have drones initially
        initial_queue_length = info.get('decision_queue_length', 0)
        
        # Current drone should be set
        current_drone = info.get('current_drone_id', -1)
        self.assertGreaterEqual(current_drone, -1)
        
        # If current drone is valid, should be < num_drones
        if current_drone >= 0:
            self.assertLess(current_drone, self.env.unwrapped.num_drones)
    
    def test_multiple_steps(self):
        """Test taking multiple steps."""
        obs, info = self.env.reset(seed=42)
        
        # Take 10 steps
        for i in range(10):
            action = i % 5  # Cycle through rules
            obs, reward, terminated, truncated, info = self.env.step(action)
            
            if terminated or truncated:
                break
        
        # Should have made some decisions
        stats = self.env.get_statistics()
        self.assertGreater(stats['total_decisions'], 0)
    
    def test_statistics(self):
        """Test statistics tracking."""
        obs, info = self.env.reset(seed=42)
        
        # Take a few steps
        for i in range(5):
            obs, reward, terminated, truncated, info = self.env.step(0)
            if terminated or truncated:
                break
        
        # Get statistics
        stats = self.env.get_statistics()
        
        # Check statistics exist
        self.assertIn('total_decisions', stats)
        self.assertIn('total_skips', stats)
        self.assertIn('queue_length', stats)
        self.assertIn('current_drone_id', stats)
        
        # Statistics should be non-negative
        self.assertGreaterEqual(stats['total_decisions'], 0)
        self.assertGreaterEqual(stats['total_skips'], 0)
        self.assertGreaterEqual(stats['queue_length'], 0)
    
    def test_local_observation_mode(self):
        """Test local observation extraction."""
        base_env = ThreeObjectiveDroneDeliveryEnv(
            grid_size=10,
            num_drones=3,
            max_orders=50,
            steps_per_hour=4,
            enable_random_events=False,
        )
        wrapped_env = EventDrivenSingleUAVWrapper(
            base_env,
            max_skip_steps=10,
            local_observation=True  # Enable local observation
        )
        
        obs, info = wrapped_env.reset(seed=42)
        
        # With local observation, should have specific structure
        # (Currently returns local obs dict)
        self.assertIsInstance(obs, dict)


if __name__ == '__main__':
    unittest.main()
