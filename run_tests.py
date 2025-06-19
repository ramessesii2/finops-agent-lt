#!/usr/bin/env python3
"""
Test runner for forecasting agent tests.
"""

import sys
import os
import subprocess
import argparse

def run_tests(test_type='all', verbose=False):
    """Run tests based on type."""
    
    # Add project root to path
    project_root = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, project_root)
    
    # Build pytest command
    cmd = ['python', '-m', 'pytest']
    
    if verbose:
        cmd.append('-v')
    
    # Add test paths based on type
    if test_type == 'unit':
        cmd.append('tests/unit/')
    elif test_type == 'integration':
        cmd.append('tests/integration/')
    elif test_type == 'e2e':
        cmd.append('tests/e2e/')
    elif test_type == 'all':
        cmd.append('tests/')
    else:
        print(f"Unknown test type: {test_type}")
        return False
    
    # Add coverage if available
    try:
        import coverage
        cmd.extend(['--cov=src', '--cov-report=html', '--cov-report=term'])
    except ImportError:
        print("Coverage not available, running tests without coverage")
    
    print(f"Running {test_type} tests...")
    print(f"Command: {' '.join(cmd)}")
    
    # Run tests
    result = subprocess.run(cmd, cwd=project_root)
    return result.returncode == 0

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Run forecasting agent tests')
    parser.add_argument('--type', choices=['unit', 'integration', 'e2e', 'all'], 
                       default='all', help='Type of tests to run')
    parser.add_argument('--verbose', '-v', action='store_true', 
                       help='Verbose output')
    
    args = parser.parse_args()
    
    success = run_tests(args.type, args.verbose)
    
    if success:
        print("✅ All tests passed!")
        sys.exit(0)
    else:
        print("❌ Some tests failed!")
        sys.exit(1)

if __name__ == '__main__':
    main() 