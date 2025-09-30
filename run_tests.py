#!/usr/bin/env python3
"""
Test runner script for Doppelganger package.

This script provides different ways to run the test suite:
- All tests
- Specific test categories
- Individual test files
"""

import sys
import os
import subprocess
import argparse

def run_tests(test_type="all", verbose=True, coverage=False):
    """
    Run pytest with specified options.
    
    Args:
        test_type (str): Type of tests to run ('all', 'unit', 'integration', 'etl', 'synth', 'audit')
        verbose (bool): Whether to run in verbose mode
        coverage (bool): Whether to generate coverage report
    """
    
    # Base pytest command
    cmd = ["python", "-m", "pytest"]
    
    # Add verbosity
    if verbose:
        cmd.append("-v")
    
    # Add coverage if requested
    if coverage:
        cmd.extend(["--cov=dpplgngr", "--cov-report=html", "--cov-report=term"])
    
    # Add test selection based on type
    if test_type == "all":
        cmd.append("test/")
    elif test_type == "unit":
        cmd.extend(["-m", "unit", "test/"])
    elif test_type == "integration":
        cmd.extend(["-m", "integration", "test/"])
    elif test_type == "etl":
        cmd.append("test/test_etl.py")
    elif test_type == "synth":
        cmd.append("test/test_synth.py")
    elif test_type == "audit":
        cmd.append("test/test_audit.py")
    elif test_type == "main":
        cmd.append("test/test.py")
    else:
        print(f"Unknown test type: {test_type}")
        return False
    
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, cwd="/home/sbenson/sw/Doppelganger")
        return result.returncode == 0
    except Exception as e:
        print(f"Error running tests: {e}")
        return False


def main():
    """Main function to handle command line arguments and run tests."""
    
    parser = argparse.ArgumentParser(description="Run Doppelganger test suite")
    parser.add_argument(
        "--type", 
        choices=["all", "unit", "integration", "etl", "synth", "audit", "main"],
        default="all",
        help="Type of tests to run"
    )
    parser.add_argument(
        "--quiet", 
        action="store_true",
        help="Run in quiet mode (less verbose output)"
    )
    parser.add_argument(
        "--coverage",
        action="store_true", 
        help="Generate coverage report"
    )
    parser.add_argument(
        "--install-deps",
        action="store_true",
        help="Install test dependencies before running tests"
    )
    
    args = parser.parse_args()
    
    # Install dependencies if requested
    if args.install_deps:
        print("Installing test dependencies...")
        deps_cmd = ["pip", "install", "pytest", "pytest-cov", "pytest-mock"]
        subprocess.run(deps_cmd)
    
    # Run tests
    success = run_tests(
        test_type=args.type,
        verbose=not args.quiet,
        coverage=args.coverage
    )
    
    if success:
        print("\n✅ All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
