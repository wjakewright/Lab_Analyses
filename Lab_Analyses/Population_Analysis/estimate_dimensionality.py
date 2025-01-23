import numpy as np
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.preprocessing import StandardScaler

from Lab_Analyses.Utilities import data_utilities as d_utils
from Lab_Analyses.Utilities.activity_timestamps import (
    get_activity_timestamps,
    refine_activity_timestamps,
)


def estimate_dimensionality_pca(
    dFoF,
    lever_active,
    cutoff=0.90,
    activity_window=(-0.5, 2),
    sampling_rate=60,
    n_comps=40,
):
    """Function to estimate the dimensionality of a neural population activity during movements
        using PCA

    INPUT PARAMETERS
        dFoF - 2d np.array of the dFoF traces for each ROI, Columns represent
                each ROI.

        lever_active - 1d binary np.ndarray of the lever active trace

        cutoff - float specifying the percent variance explained to use for the
                cutoff

        activity_window - tuple specifying the time window around movements to
                            analyze in sec

        sampling_rate - int specifying the imaging rate

        n_comps - int specifying how many components to fit to the PCA

    OUTPUT PARAMETERS
        dimensionality - int specifying the estimated dimensionality of the activity

        cum_variance - np.array of the cummulative variance explained by each component

        variance_explained - np.array of the variance explained ratio for each component


    """
    if dFoF.shape[1] < n_comps:
        n_comps = dFoF.shape[1]

    scaler = StandardScaler()

    # Get and organize data around lever presses
    ## Get movement onsets
    timestamps = get_activity_timestamps(lever_active)
    timestamps = refine_activity_timestamps(
        timestamps,
        window=activity_window,
        max_len=dFoF.shape[0],
        sampling_rate=sampling_rate,
    )
    timestamps = [x[0] for x in timestamps]

    ## Get movement-aligned activity
    ROI_ids = [f"ROI_{i}" for i in range(dFoF.shape[1])]

    _, mean_traces = d_utils.get_trace_mean_sem(
        dFoF, ROI_ids, timestamps, activity_window, sampling_rate
    )

    mean_traces = [x[0] for x in mean_traces.values()]
    mean_traces = np.vstack(mean_traces).T

    # Normalize the data
    ## Fit scaler to mean data
    scaler.fit(mean_traces)
    ## Transform the mean traces
    norm_mean_traces = scaler.transform(mean_traces)

    norm_mean_traces = mean_traces

    # Perform PCA on mean traces
    pca_model = PCA(n_components=n_comps, svd_solver="full")
    ## Fit the model
    pca_model.fit(norm_mean_traces)

    # Get the variance explained by each component
    variance_explained = pca_model.explained_variance_ratio_

    # Get cummualtive variance explained
    cum_variance = np.cumsum(variance_explained)

    # Find the number of components explaining the cutoff of variance
    dimensionality = np.argmax(cum_variance >= cutoff) + 1

    return dimensionality, cum_variance, variance_explained


def estimate_dimensionality_fa(
    dFoF,
    lever_active,
    cutoff=0.90,
    activity_window=(-0.5, 2),
    sampling_rate=60,
    n_comps=40,
):
    """Function to estimate the dimensionality of a neural population activity during movements
        using PCA

    INPUT PARAMETERS
        dFoF - 2d np.array of the dFoF traces for each ROI, Columns represent
                each ROI.

        lever_active - 1d binary np.ndarray of the lever active trace

        cutoff - float specifying the percent variance explained to use for the
                cutoff

        activity_window - tuple specifying the time window around movements to
                            analyze in sec

        sampling_rate - int specifying the imaging rate

        n_comps - int specifying how many components to fit to the PCA

    OUTPUT PARAMETERS
        dimensionality - int specifying the estimated dimensionality of the activity

        cum_variance - np.array of the cummulative variance explained by each component

        variance_explained - np.array of the variance explained ratio for each component


    """
    if dFoF.shape[1] < n_comps:
        n_comps = dFoF.shape[1]

    scaler = StandardScaler()

    # Get and organize data around lever presses
    ## Get movement onsets
    timestamps = get_activity_timestamps(lever_active)
    timestamps = refine_activity_timestamps(
        timestamps,
        window=activity_window,
        max_len=dFoF.shape[0],
        sampling_rate=sampling_rate,
    )
    timestamps = [x[0] for x in timestamps]

    ## Get movement-aligned activity
    ROI_ids = [f"ROI_{i}" for i in range(dFoF.shape[1])]

    _, mean_traces = d_utils.get_trace_mean_sem(
        dFoF, ROI_ids, timestamps, activity_window, sampling_rate
    )

    mean_traces = [x[0] for x in mean_traces.values()]
    mean_traces = np.vstack(mean_traces).T

    # Normalize the data
    ## Fit scaler to mean data
    scaler.fit(mean_traces)
    ## Transform the mean traces
    norm_mean_traces = scaler.transform(mean_traces)

    norm_mean_traces = mean_traces

    # Perform PCA on mean traces
    fa_model = FactorAnalysis(n_components=n_comps, svd_method="lapack")
    ## Fit the model
    fa_model.fit(norm_mean_traces)

    # Get the variance explained by each component
    total_var = np.nansum(np.var(norm_mean_traces, axis=0))
    variance_exp = np.nansum(fa_model.components_.T**2, axis=0)
    variance_explained = variance_exp / total_var

    # Get cummualtive variance explained
    cum_variance = np.cumsum(variance_explained)

    # Find the number of components explaining the cutoff of variance
    dimensionality = np.argmax(cum_variance >= cutoff) + 1

    return dimensionality, cum_variance, variance_explained
