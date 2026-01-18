import sys
import os

# Add project root to Python path
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import logging

import numpy as np
import matplotlib

from matplotlib import pyplot as plt

import calculator.endfield.user


def set_logger():
    logging.basicConfig(level=logging.INFO)


if __name__ == "__main__":
    calculator.endfield.user.show_graph()
