import numpy as np
import scipy.signal as sysignal
from scipy import stats

from Lab_Analyses.Utilities import data_utilities as d_utils


def synaptic_opto_responsive(dFoF, timestamps, window, sampling_rate, smooth=False):
    """Function to determine if synapses are being activated by optogenetic stimulation

    INPUT PARAMETERS
        dFoF - 2d array of the dFoF, with each column representing a spine

        timestamps - list or array of the stim onsets

        window - tuple specifying the window to analyze the activity around

        sampling_rate - int specifying the imaging rate

        smooth - boolean specifying whether to smooth the data or not

    OUTPUT PARAMETERS
        diffs - np.array of the average difference in activity for each spine

        pvalue - np.array of the pvalues for each spine

        rank - np.array of the rank values for each spine

        sig - boolean array of whether or not each spine is significantly activated


    """
    ALPHA = 0.01
    DISTANCE = 0.5 * sampling_rate
    CENTER_POINT = int(np.absolute(window[0]) * sampling_rate)
    BEFORE = int(0.5 * sampling_rate)
    AFTER = int(0.5 * sampling_rate)
    AVG_RANGE = int(0.5 * sampling_rate)

    # Initialize outputs
    diffs = np.zeros(dFoF.shape[1])
    pvalues = np.zeros(dFoF.shape[1])
    ranks = np.zeros(dFoF.shape[1])
    sigs = np.zeros(dFoF.shape[1])

    # Analyze each spine seperately
    for spine in range(dFoF.shape[1]):
        activity = dFoF[:, spine]
        # Get the activity around each stim
        traces, _ = d_utils.get_trace_mean_sem(
            activity=activity.reshape(-1, 1),
            ROI_ids=["Spine"],
            timestamps=timestamps,
            window=window,
            sampling_rate=sampling_rate,
        )
        traces = traces["Spine"]

        # Get values for each stim event
        baseline_values = []
        stim_values = []
        for event in range(traces.shape[1]):
            event_trace = traces[:, event]
            if smooth:
                event_trace = sysignal.savgol_filter(event_trace, 31, 3)
            # Get baseline value
            # baseline = np.nanmedian(
            #    event_trace[CENTER_POINT - BEFORE : CENTER_POINT]
            # )
            baseline = np.nanmedian(event_trace[0:CENTER_POINT])
            # Get the post stimulation values
            ## Get max value after stim
            stim_peak = np.argmax(event_trace[CENTER_POINT : CENTER_POINT + AFTER])
            ## Average around that point
            stim_amp = np.nanmean(
                event_trace[
                    stim_peak
                    + CENTER_POINT
                    - AVG_RANGE : stim_peak
                    + CENTER_POINT
                    + AVG_RANGE
                ]
            )
            stim_amp = np.nanmean(
                event_trace[CENTER_POINT + 10 : CENTER_POINT + AFTER + 10]
            )
            baseline_values.append(baseline)
            stim_values.append(stim_amp)
        # Perform Wilcoxon signed-rank test
        trial_diffs = np.array(stim_values) - np.array(baseline_values)
        diff = np.nanmean(np.array(stim_values) - np.array(baseline_values))
        rank, pval = stats.wilcoxon(baseline_values, stim_values)

        consistancy = np.nanmean(trial_diffs >= 0.1)

        # Assess significance
        sig = ((pval < ALPHA) and (consistancy >= 0.5)) * 1
        # sig = (pval < ALPHA) * 1
        diffs[spine] = diff
        pvalues[spine] = pval
        ranks[spine] = rank
        sigs[spine] = sig

    return diffs, pvalues, ranks, sigs
