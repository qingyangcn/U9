"""
Candidate Generator for UAV Order Selection

This module provides candidate generation strategies for the layered UAV decision system.
The upper layer generates candidate order sets for each drone, and the lower layer
applies rules to select from these candidates.

Classes:
    CandidateGenerator: Base class for candidate generation
    NearestCandidateGenerator: Generates candidates based on distance
    EarliestDeadlineCandidateGenerator: Generates candidates based on deadline
    MixedHeuristicCandidateGenerator: Combines distance and deadline
    PSOMOPSOCandidateGenerator: Placeholder for future PSO/MOPSO integration
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional, TYPE_CHECKING
import numpy as np
import math

if TYPE_CHECKING:
    from UAV_ENVIRONMENT_10 import ThreeObjectiveDroneDeliveryEnv


class CandidateGenerator(ABC):
    """
    Base class for candidate generation strategies.

    Subclasses should implement generate_candidates() to produce
    a dictionary mapping drone_id to a list of candidate order_ids.
    """

    def __init__(self, candidate_k: int = 20):
        """
        Initialize candidate generator.

        Args:
            candidate_k: Number of candidates to generate per drone
        """
        self.candidate_k = candidate_k

    @abstractmethod
    def generate_candidates(
            self,
            env: 'ThreeObjectiveDroneDeliveryEnv'
    ) -> Dict[int, List[int]]:
        """
        Generate candidate order sets for all drones.

        Args:
            env: The UAV environment instance

        Returns:
            Dictionary mapping drone_id to list of order_ids (candidates)
            Each list should contain up to candidate_k order_ids
        """
        pass

    def _get_active_orders(self, env: 'ThreeObjectiveDroneDeliveryEnv') -> List[int]:
        """Helper to get list of active order IDs from environment."""
        return list(env.active_orders)

    def _calculate_distance(self, loc1: Tuple[float, float], loc2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two locations."""
        return math.sqrt((loc1[0] - loc2[0]) ** 2 + (loc1[1] - loc2[1]) ** 2)


class NearestCandidateGenerator(CandidateGenerator):
    """Generate candidates based on nearest pickup distance."""

    def generate_candidates(
            self,
            env: 'ThreeObjectiveDroneDeliveryEnv'
    ) -> Dict[int, List[int]]:
        """
        For each drone, select K nearest orders by pickup location distance.

        Args:
            env: The UAV environment instance

        Returns:
            Dictionary mapping drone_id to list of nearest order_ids
        """
        candidates = {}
        active_orders = self._get_active_orders(env)

        for drone_id in range(env.num_drones):
            drone = env.drones[drone_id]
            drone_loc = drone['location']

            # Calculate distances to all active orders
            order_distances = []
            for order_id in active_orders:
                if order_id not in env.orders:
                    continue
                order = env.orders[order_id]
                merchant_loc = order['merchant_location']
                distance = self._calculate_distance(drone_loc, merchant_loc)
                order_distances.append((order_id, distance))

            # Sort by distance and take top K
            order_distances.sort(key=lambda x: x[1])
            candidates[drone_id] = [oid for oid, _ in order_distances[:self.candidate_k]]

        return candidates


class EarliestDeadlineCandidateGenerator(CandidateGenerator):
    """Generate candidates based on earliest delivery deadline."""

    def generate_candidates(
            self,
            env: 'ThreeObjectiveDroneDeliveryEnv'
    ) -> Dict[int, List[int]]:
        """
        For each drone, select K orders with earliest deadlines.

        Args:
            env: The UAV environment instance

        Returns:
            Dictionary mapping drone_id to list of order_ids with earliest deadlines
        """
        candidates = {}
        active_orders = self._get_active_orders(env)

        # Calculate deadlines for all active orders
        order_deadlines = []
        for order_id in active_orders:
            if order_id not in env.orders:
                continue
            order = env.orders[order_id]
            deadline = env._get_delivery_deadline_step(order)
            order_deadlines.append((order_id, deadline))

        # Sort by deadline
        order_deadlines.sort(key=lambda x: x[1])

        # Assign same candidate list to all drones
        # (each drone gets orders with earliest deadlines)
        top_k_orders = [oid for oid, _ in order_deadlines[:self.candidate_k]]
        for drone_id in range(env.num_drones):
            candidates[drone_id] = top_k_orders.copy()

        return candidates


class MixedHeuristicCandidateGenerator(CandidateGenerator):
    """
    Generate candidates using mixed heuristic (distance + deadline).

    This combines distance and deadline using a weighted score.
    """

    def __init__(
            self,
            candidate_k: int = 20,
            distance_weight: float = 0.5,
            deadline_weight: float = 0.5
    ):
        """
        Initialize mixed heuristic generator.

        Args:
            candidate_k: Number of candidates per drone
            distance_weight: Weight for distance component (0-1)
            deadline_weight: Weight for deadline component (0-1)
        """
        super().__init__(candidate_k)
        self.distance_weight = distance_weight
        self.deadline_weight = deadline_weight

    def generate_candidates(
            self,
            env: 'ThreeObjectiveDroneDeliveryEnv'
    ) -> Dict[int, List[int]]:
        """
        For each drone, select K orders based on weighted distance and deadline.

        Score = distance_weight * normalized_distance + deadline_weight * normalized_slack
        Lower score is better.

        Args:
            env: The UAV environment instance

        Returns:
            Dictionary mapping drone_id to list of order_ids
        """
        candidates = {}
        active_orders = self._get_active_orders(env)
        current_step = env.time_system.current_step

        for drone_id in range(env.num_drones):
            drone = env.drones[drone_id]
            drone_loc = drone['location']

            # Calculate scores for all active orders
            order_scores = []

            # First pass: collect raw values for normalization
            distances = []
            slacks = []
            for order_id in active_orders:
                if order_id not in env.orders:
                    continue
                order = env.orders[order_id]
                merchant_loc = order['merchant_location']
                distance = self._calculate_distance(drone_loc, merchant_loc)
                deadline = env._get_delivery_deadline_step(order)
                slack = deadline - current_step

                distances.append(distance)
                slacks.append(slack)

            if not distances:
                candidates[drone_id] = []
                continue

            # Normalize
            max_dist = max(distances) if distances else 1.0
            max_slack = max(slacks) if slacks else 1.0
            min_dist = min(distances) if distances else 0.0
            min_slack = min(slacks) if slacks else 0.0

            dist_range = max_dist - min_dist if max_dist > min_dist else 1.0
            slack_range = max_slack - min_slack if max_slack > min_slack else 1.0

            # Second pass: calculate normalized scores
            idx = 0
            for order_id in active_orders:
                if order_id not in env.orders:
                    continue

                # Normalize distance (0-1, lower is better)
                norm_distance = (distances[idx] - min_dist) / dist_range

                # Normalize slack (0-1, higher slack is worse for urgency)
                # Invert so that lower slack (more urgent) gets higher priority
                norm_urgency = 1.0 - ((slacks[idx] - min_slack) / slack_range)

                # Combined score (lower is better)
                score = (self.distance_weight * norm_distance +
                         self.deadline_weight * norm_urgency)

                order_scores.append((order_id, score))
                idx += 1

            # Sort by score and take top K
            order_scores.sort(key=lambda x: x[1])
            candidates[drone_id] = [oid for oid, _ in order_scores[:self.candidate_k]]

        return candidates


class PSOMOPSOCandidateGenerator(CandidateGenerator):
    """
    Placeholder for PSO/MOPSO-based candidate generation.

    This class provides an interface for future integration with
    Particle Swarm Optimization (PSO) or Multi-Objective PSO (MOPSO)
    algorithms for more sophisticated candidate generation.

    Current implementation falls back to mixed heuristic.
    """

    def __init__(
            self,
            candidate_k: int = 20,
            pso_params: Optional[Dict] = None
    ):
        """
        Initialize PSO/MOPSO generator.

        Args:
            candidate_k: Number of candidates per drone
            pso_params: Dictionary of PSO parameters (for future use)
        """
        super().__init__(candidate_k)
        self.pso_params = pso_params or {}
        # Fallback to mixed heuristic for now
        self._fallback = MixedHeuristicCandidateGenerator(candidate_k)

    def generate_candidates(
            self,
            env: 'ThreeObjectiveDroneDeliveryEnv'
    ) -> Dict[int, List[int]]:
        """
        Generate candidates using PSO/MOPSO (currently falls back to mixed heuristic).

        TODO: Implement actual PSO/MOPSO algorithm

        Args:
            env: The UAV environment instance

        Returns:
            Dictionary mapping drone_id to list of order_ids
        """
        # For now, use mixed heuristic as fallback
        # Future: Implement PSO/MOPSO optimization here
        return self._fallback.generate_candidates(env)