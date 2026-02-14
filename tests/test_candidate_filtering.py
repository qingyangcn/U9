"""
Unit tests for candidate filtering functionality.

Tests that rule methods respect candidate constraints when filtering is enabled.
"""

import unittest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from UAV_ENVIRONMENT_9 import ThreeObjectiveDroneDeliveryEnv, OrderStatus
from candidate_generator import NearestCandidateGenerator, EarliestDeadlineCandidateGenerator, MixedHeuristicCandidateGenerator


class TestCandidateFiltering(unittest.TestCase):
    """Test suite for candidate-based filtering."""
    
    def setUp(self):
        """Create a minimal environment for testing."""
        self.env = ThreeObjectiveDroneDeliveryEnv(
            grid_size=10,
            num_drones=3,
            max_orders=50,
            steps_per_hour=4,
            enable_random_events=False,
            candidate_fallback_enabled=True,
            candidate_update_interval=0,  # Manual update only
        )
    
    def test_candidate_generator_nearest(self):
        """Test nearest candidate generator."""
        # Reset environment
        self.env.reset(seed=42)
        
        # Create and set nearest candidate generator
        generator = NearestCandidateGenerator(candidate_k=5)
        self.env.set_candidate_generator(generator)
        
        # Update candidates
        self.env.update_filtered_candidates()
        
        # Check that candidates were generated
        self.assertIsNotNone(self.env.filtered_candidates)
        
        # Each drone should have candidates
        for drone_id in range(self.env.num_drones):
            candidates = self.env.get_filtered_candidates_for_drone(drone_id)
            self.assertIsInstance(candidates, list)
            # May have fewer than K if not enough orders
            self.assertLessEqual(len(candidates), 5)
    
    def test_candidate_generator_earliest_deadline(self):
        """Test earliest deadline candidate generator."""
        self.env.reset(seed=42)
        
        generator = EarliestDeadlineCandidateGenerator(candidate_k=5)
        self.env.set_candidate_generator(generator)
        self.env.update_filtered_candidates()
        
        # Check candidates exist
        for drone_id in range(self.env.num_drones):
            candidates = self.env.get_filtered_candidates_for_drone(drone_id)
            self.assertIsInstance(candidates, list)
    
    def test_candidate_generator_mixed(self):
        """Test mixed heuristic candidate generator."""
        self.env.reset(seed=42)
        
        generator = MixedHeuristicCandidateGenerator(
            candidate_k=5,
            distance_weight=0.5,
            deadline_weight=0.5
        )
        self.env.set_candidate_generator(generator)
        self.env.update_filtered_candidates()
        
        # Check candidates exist
        for drone_id in range(self.env.num_drones):
            candidates = self.env.get_filtered_candidates_for_drone(drone_id)
            self.assertIsInstance(candidates, list)
    
    def test_rule_respects_candidates(self):
        """Test that rules respect candidate constraints."""
        self.env.reset(seed=42)
        
        # Create a restricted candidate set for drone 0
        # Only allow first 2 active orders
        active_orders_list = list(self.env.active_orders)
        if len(active_orders_list) >= 2:
            restricted_candidates = {
                0: active_orders_list[:2],
                1: active_orders_list,
                2: active_orders_list,
            }
            self.env.filtered_candidates = restricted_candidates
            
            # Try to select order using rule for drone 0
            # Should only select from restricted candidates
            selected_order = self.env._rule_ready_edf(0)
            
            # If an order was selected, it should be in the candidate set
            if selected_order is not None:
                self.assertIn(selected_order, restricted_candidates[0])
    
    def test_fallback_when_no_candidates(self):
        """Test fallback to active_orders when candidates are empty."""
        self.env.reset(seed=42)
        
        # Set empty candidates for drone 0
        self.env.filtered_candidates = {
            0: [],  # Empty candidates
            1: list(self.env.active_orders),
            2: list(self.env.active_orders),
        }
        
        # With fallback enabled, should still select from active_orders
        self.env.candidate_fallback_enabled = True
        selected_order = self.env._rule_ready_edf(0)
        
        # May or may not select an order depending on state,
        # but should not raise an error
        self.assertTrue(selected_order is None or selected_order in self.env.orders)
    
    def test_no_fallback_when_disabled(self):
        """Test that fallback can be disabled."""
        self.env.reset(seed=42)
        
        # Set empty candidates for drone 0
        self.env.filtered_candidates = {
            0: [],  # Empty candidates
            1: list(self.env.active_orders),
            2: list(self.env.active_orders),
        }
        
        # Disable fallback
        self.env.candidate_fallback_enabled = False
        selected_order = self.env._rule_ready_edf(0)
        
        # Should not select any order since candidates are empty
        self.assertIsNone(selected_order)
    
    def test_get_decision_drones(self):
        """Test get_decision_drones method."""
        self.env.reset(seed=42)
        
        # Get drones at decision points
        decision_drones = self.env.get_decision_drones()
        
        # Should return a list
        self.assertIsInstance(decision_drones, list)
        
        # All IDs should be valid drone IDs
        for drone_id in decision_drones:
            self.assertGreaterEqual(drone_id, 0)
            self.assertLess(drone_id, self.env.num_drones)
    
    def test_apply_rule_to_drone(self):
        """Test apply_rule_to_drone method."""
        self.env.reset(seed=42)
        
        # Get a drone at decision point
        decision_drones = self.env.get_decision_drones()
        
        if decision_drones:
            drone_id = decision_drones[0]
            
            # Apply rule 2 (READY_EDF)
            result = self.env.apply_rule_to_drone(drone_id, rule_id=2)
            
            # Result should be boolean
            self.assertIsInstance(result, bool)
    
    def test_candidate_update_on_reset(self):
        """Test that candidates are updated on reset."""
        # Set a generator
        generator = NearestCandidateGenerator(candidate_k=5)
        self.env.set_candidate_generator(generator)
        
        # Reset should update candidates
        self.env.reset(seed=42)
        
        # Check that candidates exist
        self.assertIsNotNone(self.env.filtered_candidates)
        self.assertTrue(len(self.env.filtered_candidates) > 0)
    
    def test_candidate_constrained_orders(self):
        """Test _get_candidate_constrained_orders method."""
        self.env.reset(seed=42)
        
        # Create test candidates
        all_orders = list(self.env.active_orders)
        if len(all_orders) >= 5:
            test_candidates = all_orders[:3]
            self.env.filtered_candidates = {
                0: test_candidates,
            }
            
            # Get constrained orders
            constrained = self.env._get_candidate_constrained_orders(0, all_orders)
            
            # Should only include candidates
            self.assertEqual(set(constrained), set(test_candidates))
            
            # With fallback disabled and empty candidates, should return empty
            self.env.filtered_candidates[0] = []
            self.env.candidate_fallback_enabled = False
            constrained = self.env._get_candidate_constrained_orders(0, all_orders)
            self.assertEqual(len(constrained), 0)
            
            # With fallback enabled and empty candidates, should return all
            self.env.candidate_fallback_enabled = True
            constrained = self.env._get_candidate_constrained_orders(0, all_orders)
            self.assertEqual(set(constrained), set(all_orders))


if __name__ == '__main__':
    unittest.main()
