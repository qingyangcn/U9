"""
Basic structure validation for U10 decentralized execution components.

This script validates that all modules can be imported and have the expected structure.
It does NOT run the full environment (which requires data files and dependencies).
"""

import sys
import os

# Add repo root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_imports():
    """Test that all modules can be imported."""
    print("Testing module imports...")
    
    try:
        # Test importing the new modules
        print("  - Importing U10_decentralized_execution...")
        import U10_decentralized_execution
        assert hasattr(U10_decentralized_execution, 'DecentralizedEventDrivenExecutor')
        
        print("  - Importing U10_single_uav_training_wrapper...")
        import U10_single_uav_training_wrapper
        assert hasattr(U10_single_uav_training_wrapper, 'SingleUAVTrainingWrapper')
        
        print("  ✓ All module imports successful")
        return True
    except Exception as e:
        print(f"  ✗ Import failed: {e}")
        return False


def test_class_structure():
    """Test that classes have expected methods."""
    print("\nTesting class structure...")
    
    try:
        from U10_decentralized_execution import DecentralizedEventDrivenExecutor
        from U10_single_uav_training_wrapper import SingleUAVTrainingWrapper
        
        # Check DecentralizedEventDrivenExecutor methods
        print("  - Checking DecentralizedEventDrivenExecutor...")
        required_methods = ['reset', 'step', 'run_episode', 'get_statistics', 
                           '_process_decision_round', '_skip_to_next_decision',
                           '_extract_local_observation']
        for method in required_methods:
            assert hasattr(DecentralizedEventDrivenExecutor, method), \
                f"Missing method: {method}"
        
        # Check SingleUAVTrainingWrapper methods
        print("  - Checking SingleUAVTrainingWrapper...")
        required_methods = ['reset', 'step', '_populate_decision_queue',
                           '_skip_to_next_decision', '_extract_local_observation']
        for method in required_methods:
            assert hasattr(SingleUAVTrainingWrapper, method), \
                f"Missing method: {method}"
        
        print("  ✓ All required methods present")
        return True
    except Exception as e:
        print(f"  ✗ Structure test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_script_structure():
    """Test that training script has expected structure."""
    print("\nTesting training script structure...")
    
    try:
        # Check that U10_train.py exists and has expected content
        with open('U10_train.py', 'r') as f:
            content = f.read()
        
        # Check for key components
        assert 'training_mode' in content, "Missing training_mode parameter"
        assert 'event_driven_shared_policy' in content, "Missing event_driven_shared_policy mode"
        assert 'SingleUAVTrainingWrapper' in content, "Missing SingleUAVTrainingWrapper import"
        assert 'drone_sampling' in content, "Missing drone_sampling parameter"
        
        print("  ✓ Training script structure validated")
        return True
    except Exception as e:
        print(f"  ✗ Training script test failed: {e}")
        return False


def test_documentation():
    """Test that documentation exists."""
    print("\nTesting documentation...")
    
    try:
        # Check that U10_README.md exists
        assert os.path.exists('U10_README.md'), "U10_README.md not found"
        
        with open('U10_README.md', 'r') as f:
            content = f.read()
        
        # Check for key sections
        assert 'CTDE' in content, "Missing CTDE explanation"
        assert 'Centralized Training, Decentralized Execution' in content
        assert 'Event-Driven' in content, "Missing event-driven explanation"
        assert 'Centralized Arbitration' in content, "Missing arbitration explanation"
        
        print("  ✓ Documentation validated")
        return True
    except Exception as e:
        print(f"  ✗ Documentation test failed: {e}")
        return False


def test_sanity_check_exists():
    """Test that sanity check script exists."""
    print("\nTesting sanity check script...")
    
    try:
        assert os.path.exists('U10_sanity_check_decentralized.py'), \
            "U10_sanity_check_decentralized.py not found"
        
        with open('U10_sanity_check_decentralized.py', 'r') as f:
            content = f.read()
        
        assert 'DecentralizedEventDrivenExecutor' in content
        assert 'random_policy' in content
        assert 'run_sanity_check' in content
        
        print("  ✓ Sanity check script validated")
        return True
    except Exception as e:
        print(f"  ✗ Sanity check test failed: {e}")
        return False


def main():
    """Run all validation tests."""
    print("=" * 60)
    print("U10 Decentralized Execution - Structure Validation")
    print("=" * 60)
    
    all_passed = True
    
    # Run tests
    all_passed &= test_imports()
    all_passed &= test_class_structure()
    all_passed &= test_training_script_structure()
    all_passed &= test_documentation()
    all_passed &= test_sanity_check_exists()
    
    # Summary
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ ALL VALIDATION TESTS PASSED")
        print("=" * 60)
        print("\nNext Steps:")
        print("1. Install dependencies: pip install -r requirements_u10.txt")
        print("2. Run sanity check: python U10_sanity_check_decentralized.py")
        print("3. Start training: python U10_train.py --total-steps 1000")
        return 0
    else:
        print("✗ SOME VALIDATION TESTS FAILED")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
