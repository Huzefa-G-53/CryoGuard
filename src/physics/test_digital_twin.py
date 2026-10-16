"""
Test script for the Digital Twin implementation (F2) at root src.
"""
import os
import sys
from pathlib import Path

# Delegate to cryoguard implementation
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from cryoguard.src.physics.test_digital_twin import test_on_brand_a_sample

if __name__ == "__main__":
    success = test_on_brand_a_sample()
    if not success:
        sys.exit(1)