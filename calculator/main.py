import logging

import numpy as np
import matplotlib

from matplotlib import pyplot as plt


def set_logger():
    logging.basicConfig(level=logging.INFO)


if __name__ == "__main__":
    print("Matplotlib plt backend: {}".format(plt.get_backend()))
    x = np.linspace(0, 10, 30)
    y = np.sin(x)
    plt.scatter(x, y)
    plt.show()
