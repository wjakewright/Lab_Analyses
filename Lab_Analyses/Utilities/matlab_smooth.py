import numpy as np


def matlab_smooth(data, window):
    """Helper function to replication the implementation of matlab smooth function
    
        data = 1d np.array
        
        window = int. Must be odd value
    """
    if window % 2 == 0:
        window = window + 1
    out0 = np.convolve(data, np.ones(window, dtype=int), "valid") / window
    r = np.arange(1, window - 1, 2)
    start = np.cumsum(data[: window - 1])[::2] / r
    stop = (np.cumsum(data[:-window:-1])[::2] / r)[::1]

    return np.concatenate((start, out0, stop))
