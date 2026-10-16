"""
Digital Twin (Grey-Box) Model for Refrigeration Unit Health Monitoring.

Synchronized wrapper / implementation for root src package.
"""

from cryoguard.src.physics.digital_twin import DigitalTwin, add_digital_twin_features

__all__ = ['DigitalTwin', 'add_digital_twin_features']

if __name__ == "__main__":
    print("Digital Twin module for CryoGuard AI")