"""Module for processing raw fluorescence traces to calculate deltaF/F. 
   Utilizes a drifting baseline across the entire session"""

import numpy as np
from Lab_Analyses.Utilities.baseline_kde import baseline_kde
from Lab_Analyses.Utilities.matlab_smooth import matlab_smooth


def get_dFoF(
    data, sampling_rate, smooth_window=0.5, bout_separations=None, artifact_frames=None
):
    """Function to calculate dFoF from raw fluorescence traces
        
        INPUT PARAMETERS
            data - 1d array of fluorescence trace of a single roi.

            smooth_window - float indicating over what time window to smooth the trace
                            Optional with default set to 0.5 seconds
            
            sample_rate - float indicating the imaging sampling rate. Default set to 30Hz

            bout_separations - list indicating which frames are the start of a new imaging bout.
                                e.g. when performing imaging loops
            
            artifact_frames - list of tuples indicating frames to blank due to significant 
                                motion artifacts that are not corrected (e.g. manual z correction)
        
        OUTPUT PARAMETERS
            dFoF - array of the dFoF trace for the roi. 

            processed_dFoF - array of the smooth dFoF trace for the roi

            drifting_baseline - array of the estimated drifting baseline used to 
                                calulate dFoF
    """
    # Constants
    DS_RATIO = 20
    WINDOW = np.round(sampling_rate)
    STEP = 20
    SECONDS_TO_IGNORE = 10  # only used to correct for bout separations
    PAD_LENGTH = 1000
    SMOOTH_PAD_LENGTH = 500
    SMOOTH_WINDOW = int(smooth_window * np.round(sampling_rate))

    if artifact_frames is not None:
        jump_correction = True
    else:
        jump_correction = False

    # Fix NaN values for smoothing
    if np.isnan(data).any():
        # Get indecies of NaN values
        nan_inds = np.nonzero(np.isnan(data))[0]
        # Check if there is something wrong with the start of the trace
        first_val = np.nonzero(~np.isnan(data))[0][0]
        # Replace missing frames with baseline estimation
        if first_val > 10:
            data[:first_val] = baseline_kde(
                data[first_val : first_val + first_val - 1],
                ds_ratio=DS_RATIO,
                window=WINDOW,
                step=STEP,
            )
        # Get rid of NaN values. Mimic noisy trace
        data[nan_inds] = np.nanmedian(data) * np.ones(
            np.sum(np.isnan(data))
        ) + np.nanstd(data) * np.random.randn(np.sum(np.isnan(data)))

    # Fix data near zero
    if np.any(data < 1):
        data = data + np.absolute(np.nanmin(data))

    ## Pad data to prevent edge effects when estimating the baseline

    # Bounded curve that roughly estimates the baseline which will be used to padd data
    est_base = baseline_kde(data, DS_RATIO, WINDOW, STEP)

    # Remove start frames between imaging bouts (removes z shifts and photoactivation artifacts)
    if bout_separations is not None:
        start_frames = np.array([0] + bout_separations)
        blank_windows = start_frames + np.ceil(SECONDS_TO_IGNORE * sampling_rate)

        for start, ignore in zip(start_frames, blank_windows):
            if start > len(data):
                continue
            if ignore > len(data):
                ignore = len(data)
            data[start:ignore] = est_base[start:ignore] + np.nanstd(
                data
            ) * np.random.randn(len(np.arange(start, ignore)))

    # Correct for large uncorrected motion artifacts
    if jump_correction is True:
        for artifact in artifact_frames:
            start = artifact[0]
            end = artifact[1]
            data[start:end] = est_base[start:end] + (
                0.5 * np.nanstd(data[start:end])
            ) * np.random.randn(len(np.arange(start, end)))

    # Generate baseline
    pad_start = est_base[np.random.randint(low=0, high=1000, size=PAD_LENGTH)]
    pad_end = est_base[
        np.random.randint(
            low=len(est_base) - PAD_LENGTH, high=len(est_base), size=PAD_LENGTH
        )
    ]
    padded_data = np.concatenate((pad_start, data, pad_end))

    # Kernel Density Estimation (Aki's method)
    true_baseline_kde = baseline_kde(padded_data, DS_RATIO, WINDOW, STEP)
    drifting_baseline = true_baseline_kde[PAD_LENGTH:-PAD_LENGTH]

    # Baseline subtraction
    bl_sub = data - drifting_baseline

    # Baseline Division (since using raw fluorescence traces)
    if np.nanmedian(drifting_baseline) != 0:
        dFoF = bl_sub / drifting_baseline
    else:
        bl_sub = bl_sub + 1
        drifting_baseline = drifting_baseline + 1
        dFoF = bl_sub / drifting_baseline

    # Smooth the calculated dFoF
    pad_start = np.nanstd(dFoF) * np.random.randn(SMOOTH_PAD_LENGTH)
    pad_end = np.nanstd(dFoF) * np.random.randn(SMOOTH_PAD_LENGTH)

    padded_dFoF = np.concatenate((pad_start, dFoF, pad_end))
    padded_smoothed = matlab_smooth(padded_dFoF, SMOOTH_WINDOW)

    processed_dFoF = padded_smoothed[SMOOTH_PAD_LENGTH:-SMOOTH_PAD_LENGTH]

    return dFoF, processed_dFoF, drifting_baseline
