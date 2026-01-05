#!/usr/bin/env python3
"""
Pre-release verification script.
Run this before making the repository public.
"""

import subprocess
import sys
from pathlib import Path


def run_check(name: str, cmd: list) -> bool:
    """Run a check and return success status."""
    print(f"\n{'='*60}")
    print(f"CHECK: {name}")
    print('='*60)
    
    try:
        # Using shell=True for complex commands if needed, but here simple lists should work
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ PASSED")
            return True
        else:
            print(f"❌ FAILED")
            print(result.stdout)
            print(result.stderr)
            return False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False


def main():
    # Ensure we are in project root
    root = Path(__file__).parent.parent
    import os
    os.chdir(root)

    checks = [
        ("Rust Backend Compiles", ["maturin", "develop", "--release"]),
        ("Python Imports Work", ["python3", "-c", "from mmr_elites.algorithms import run_mmr_elites; print('OK')"]),
        ("Unit Tests Pass", ["env", "PYTHONPATH=.", "pytest", "tests/unit/", "-v", "--tb=short"]),
        ("Integration Tests Pass", ["env", "PYTHONPATH=.", "pytest", "tests/integration/", "-v", "--tb=short"]),
        ("CLI Works", ["env", "PYTHONPATH=.", "python3", "-m", "mmr_elites.cli", "--help"]),
        ("Quick Benchmark Runs", ["env", "PYTHONPATH=.", "python3", "-m", "mmr_elites.cli", "run", "--generations", "10", "--seed", "42"]),
    ]
    
    # Optional checks
    optional_checks = [
        ("Code Formatting", ["black", "--check", "mmr_elites/", "tests/"]),
        ("Import Sorting", ["isort", "--check", "mmr_elites/", "tests/"]),
    ]
    
    print("\n" + "="*60)
    print("MMR-ELITES PRE-RELEASE VERIFICATION")
    print("="*60)
    
    passed = 0
    failed = 0
    
    for name, cmd in checks:
        if run_check(name, cmd):
            passed += 1
        else:
            failed += 1
    
    print("\n" + "="*60)
    print("OPTIONAL CHECKS")
    print("="*60)
    
    for name, cmd in optional_checks:
        run_check(name, cmd)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Passed: {passed}/{len(checks)}")
    print(f"Failed: {failed}/{len(checks)}")
    
    if failed == 0:
        print("\n🎉 ALL CHECKS PASSED - Ready for release!")
        return 0
    else:
        print(f"\n⚠️  {failed} check(s) failed - Please fix before release")
        return 1


if __name__ == "__main__":
    sys.exit(main())
