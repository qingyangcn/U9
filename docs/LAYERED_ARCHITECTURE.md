# Layered Architecture Documentation

## Overview

The U9 system has been enhanced with a layered decision-making architecture that separates candidate generation (upper layer) from rule-based selection (lower layer). This document explains the architecture, implementation, and usage.

## Architecture Components

### 1. Upper Layer - Candidate Generation
- Generates candidate order sets for each drone
- Configurable strategies: Nearest, Earliest Deadline, Mixed, PSO/MOPSO
- Returns `Dict[drone_id, List[order_ids]]`

### 2. Lower Layer - Rule Selection
- Selects orders from candidate sets using rules
- 5 interpretable rules constrained by candidates
- Fallback to all orders when candidates empty (configurable)

### 3. Event-Driven Interface
- Transforms `MultiDiscrete([5]*N)` to `Discrete(5)`
- Processes drones one at a time at decision points
- Supports homogeneous policy parameter sharing

## Quick Start

See main README.md for complete usage examples and API documentation.

For detailed architecture information, refer to the code comments and docstrings in:
- `candidate_generator.py` - Candidate generation strategies
- `UAV_ENVIRONMENT_9.py` - Environment extensions
- `wrappers/event_driven_single_uav_wrapper.py` - Event-driven wrapper
