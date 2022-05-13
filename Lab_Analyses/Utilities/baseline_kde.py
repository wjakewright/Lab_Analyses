"""Estimates fluorescence baseline using kernel density estimation
    This code is translated from Aki Mitani's MatLab code"""


import numpy as np
from scipy import interpolate, stats


def baseline_kde(x, ds_ratio, window, step):
    """Estimates fluorescence baseline using kernel density estimation
    
    INPUT PARAMETERS
        x - array of fluorescence traces
        
        ds_ratio - int specifying the downsample ratio
        
        window - int specifying the number of donwsampled samples to use baseline est
        
        step - int specifying the stepsize to increase speed
        
    OUTPUT PARAMETERS
        estimated_baseline - array of the estimated baseline
        
    """
    # Ensure parity between window and step sizes
    if window % 2 != step % 2:
        window = window + 1

    # downsample data
    x_ds = downsample_mean(x, ds_ratio)
    # Get where the downsampled points lie along the original array
    i_ds = downsample_mean(np.arange(len(x)), ds_ratio)

    h = (window - step) / 2

    i_steps = []
    b_steps = []
    for i in np.arange(0, len(x_ds), step):
        r = int(np.amax(np.array([0, i - h])))
        l = int(np.amin(np.array([len(x_ds), i + step - 1 + h])))
        i_steps.append(
            np.nanmean(i_ds[i : np.amin(np.array([i + step - 1, len(x_ds)]))])
        )
        b_steps.append(mode_kde(x_ds[r:l]))

    baseline_interpolater = interpolate.interp1d(
        i_steps, b_steps, kind="cubic", fill_value="extrapolate"
    )
    estimated_baseline = baseline_interpolater(np.arange(len(x)))

    return estimated_baseline


def mode_kde(x):
    # Helper function to perform the baseline kernel density estimation
    x = x[~np.isnan(x)]
    kde = stats.gaussian_kde(x)
    pts = np.linspace(x.min(), x.max(), 200)
    f = kde(pts)
    ii = np.nanargmax(f)
    ii_1 = np.amax(np.array([ii - 1, 1]))
    ii_2 = np.amin(np.array([ii + 1, len(f)]))

    if ii_2 - ii_1 == 2:
        if f[ii_2] > f[ii_1]:
            if f[ii] - f[ii_2] < f[ii_2] - f[ii_1]:
                ii_1 = ii
        else:
            if f[ii] - f[ii_1] < f[ii_1] - f[ii_2]:
                ii_2 = ii

    xx = np.linspace(pts[ii_1], pts[ii_2], 201)
    new_f = kde(xx)
    new_ii = np.nanargmax(new_f)

    m = xx[new_ii]

    return m


def downsample_mean(x, ds_ratio):
    # Helper function to downsample the data
    end = ds_ratio * int(len(x) / ds_ratio)
    ds = np.nanmean(x[:end].reshape(-1, ds_ratio), 1)

    return ds
