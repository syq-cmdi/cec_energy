# energy_manager.py - Bridge to original code
import sys
import os

# Add the parent directory to the path so we can import test.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the original components from your test.py
from test import EdgeNode, CloudStorage, EnergyManager, visualize_results

# Export the classes that our RL code expects
__all__ = ['EdgeNode', 'CloudStorage', 'EnergyManager', 'visualize_results']