import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SIM_ENV_DIR = os.path.dirname(CURRENT_DIR)
PROJECT_ROOT = os.path.dirname(SIM_ENV_DIR)
for path in (PROJECT_ROOT, SIM_ENV_DIR):
    if path not in sys.path:
        sys.path.append(path)

from sim_env.dataset import RadarFileDataset

__all__ = ['RadarFileDataset']
