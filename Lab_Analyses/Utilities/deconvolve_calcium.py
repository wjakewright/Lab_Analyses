"""Module to handel spike deconvolution of calcium traces. This is adapted from 
    the Oasis algorithm incorporated in Suite2p"""


import numpy as np
from numba import njit, prange


def oasis(fluo, batch_size, tau, sampling_rate):
    """Computes nont-negative deconvolution of calcium signal
    
        INPUT PARAMETERS
            fluo - 2d array of preprocessed calcium traces [time x rois]
            
            batch_size - int specifying number of frames processed per batch
            
            tau - float specifying the timescale of the sensor and is
                  used for the deconvolution kernel
                  
            sampling_rate - float specifying the imaging sampling rate
            
        OUTPUT PARAMETERS
            deconvolved - 2d array of deconvolved calcium traces [time x rois]
    
    """
    # Reshape the input array
    fluo = fluo.T

    # Set up matrix shape and set datatypes
    NN, NT = fluo.shape
    fluo = fluo.astype(np.float32)
    deconvolved = np.zeros((NN, NT), dtype=np.float32)

    # Perform deconvolution in batches
    for i in range(0, NN, batch_size):
        f = fluo[i : i + batch_size]
        v = np.zeros((f.shape[0], NT), dtype=np.float32)
        w = np.zeros((f.shape[0], NT), dtype=np.float32)
        t = np.zeros((f.shape[0], NT), dtype=np.int64)
        l = np.zeros((f.shape[0], NT), dtype=np.float32)
        s = np.zeros((f.shape[0], NT), dtype=np.float32)
        oasis_matrix(f, v, w, t, l, s, tau, sampling_rate)
        deconvolved[i : i + batch_size] = s

    # Reshape output
    deconvolved = deconvolved.T

    return deconvolved


@njit(
    [
        "float32[:], float32[:], float32[:], int64[:], float32[:], float32[:], float32, float32"
    ],
    cache=True,
)
def oasis_trace(F, v, w, t, l, s, tau, fs):
    """Spike deconvolution on a single neuron"""
    NT = F.shape[0]
    g = -1.0 / (tau * fs)

    it = 0
    ip = 0

    while it < NT:
        v[ip], w[ip], t[ip], l[ip] = F[it], 1, it, 1
        while ip > 0:
            if v[ip - 1] * np.exp(g * l[ip - 1]) > v[ip]:
                # violation of the constraint means merging pools
                f1 = np.exp(g * l[ip - 1])
                f2 = np.exp(2 * g * l[ip - 1])
                wnew = w[ip - 1] + w[ip] * f2
                v[ip - 1] = (v[ip - 1] * w[ip - 1] + v[ip] * w[ip] * f1) / wnew
                w[ip - 1] = wnew
                l[ip - 1] = l[ip - 1] + l[ip]
                ip -= 1
            else:
                break
        it += 1
        ip += 1
    s[t[1:ip]] = v[1:ip] - v[: ip - 1] * np.exp(g * l[: ip - 1])


@njit(
    [
        "float32[:,:], float32[:,:], float32[:,:], int64[:,:], float32[:,:], float32[:,:], float32, float32"
    ],
    parallel=True,
    cache=True,
)
def oasis_matrix(F, v, w, t, l, s, tau, fs):
    """Spike deconvolution on many neurons parallelized with prange"""
    for n in prange(F.shape[0]):
        oasis_trace(F[n], v[n], w[n], t[n], l[n], s[n], tau, fs)

