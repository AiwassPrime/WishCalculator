import logging

import cupy as cp
import numpy as np
import time

import genshin.consts as consts


def set_logger():
    logging.basicConfig(level=logging.INFO)


if __name__ == "__main__":
    set_logger()
    numbers = [4, 2, 7, 1, 9, 3, 7, 5]

    # Number to find and move to index 0
    i = 7

    # Find the index of the first occurrence of number i
    if i in numbers:
        index_i = numbers.index(i)

        # Cut the list so that number i appears at index 0
        new_list = numbers[index_i:]
        old_list = numbers[:index_i]

        print("Original List:", numbers)
        print("Modified List:", new_list)
        print("Modified List Front:", old_list)

    else:
        print(f"Number {i} not found in the list.")
