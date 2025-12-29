#!/usr/bin/env python3
"""
Cleanup script to remove legacy code after verification.

Run this ONLY after:
1. All tests pass
2. Experiments run successfully
3. You've verified new code works correctly
"""

import shutil
from pathlib import Path


def cleanup_legacy(dry_run: bool = True):
    """Remove legacy directory."""
    legacy_dir = Path("legacy")
    
    if not legacy_dir.exists():
        print("No legacy directory found.")
        return
    
    # List files to be removed
    files = list(legacy_dir.rglob("*"))
    print(f"Found {len(files)} files in legacy/")
    
    if dry_run:
        print("\n[DRY RUN] Would remove:")
        for f in files[:20]:
            print(f"  {f}")
        if len(files) > 20:
            print(f"  ... and {len(files) - 20} more")
        print("\nRun with --execute to actually remove files.")
    else:
        # Check if we should really delete
        confirm = input("Are you sure you want to delete legacy/? [y/N] ")
        if confirm.lower() == 'y':
            shutil.rmtree(legacy_dir)
            print("✅ Legacy directory removed.")
        else:
            print("Cancelled.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="Actually delete files")
    args = parser.parse_args()
    
    cleanup_legacy(dry_run=not args.execute)
