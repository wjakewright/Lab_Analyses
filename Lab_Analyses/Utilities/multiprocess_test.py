import multiprocessing
import time

import numpy as np


def square(x, y):
    time.sleep(1)
    return x * y


if __name__ == "__main__":
    pool = multiprocessing.Pool(processes=4)
    inputs = np.random.randn(50) * 100
    inputs_2 = np.random.randn(50) * 100
    start = time.process_time()
    outputs = pool.starmap(square, zip(inputs, inputs_2))
    print(f"{outputs}")
