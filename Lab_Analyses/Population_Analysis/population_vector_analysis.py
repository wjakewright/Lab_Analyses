import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from Lab_Analyses.Behavior.mouse_lever_behavior import correlate_btw_sessions
from Lab_Analyses.Population_Analysis.population_utilities import (
    calc_pairwise_distances_btw_vectors,
)
from Lab_Analyses.Utilities import data_utilities as d_utils
from Lab_Analyses.Utilities.activity_timestamps import (
    get_activity_timestamps,
    refine_activity_timestamps,
)


def pca_population_vector_analysis(
    dFoF,
    lever_active,
    activity_window=(-2, 4),
    sampling_rate=60,
    n_comps=20,
):
    """Function to analyze the pouplation dynamics around movements
    within a single session using PCA

    INPUT PARAMTERS
        dFoF - 2d np.array of the dFoF traces for each ROI, Columns represent
                each ROI.

        lever_active - 1d binary np.ndarray of the lever active trace

        activity_window - tuple specifying the time window around movements to
                            analyze in sec

        sampling_rate - int specifying the imaging rate

    OUTPUT PARAMETERS
        avg_pop_vector - 2d np.array of the PCA reduced trial averaged activity

        event_pop_vector - list of 2d np.arrays of the PCA reduced activity for
                            each individual trial

        all_similarities - np.array of the similarity values between all vector
                            comparisons

        med_similarity - float of the median similarity for all vector comparisons


    """
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

    ind_traces, mean_traces = d_utils.get_trace_mean_sem(
        dFoF, ROI_ids, timestamps, activity_window, sampling_rate
    )
    ## Reorganize the outputs
    ind_traces = list(ind_traces.values())
    ### Need to get traces for each event rather than for each roi
    event_traces = []
    for i, _ in enumerate(timestamps):
        temp_traces = [x[:, i] for x in ind_traces]
        event_traces.append(np.vstack(temp_traces).T)
    mean_traces = [x[0] for x in mean_traces.values()]
    mean_traces = np.vstack(mean_traces).T

    # Normalize the data
    ## Fit scaler to mean data
    scaler.fit(mean_traces)
    ## Transform the mean traces
    norm_mean_traces = scaler.transform(mean_traces)
    ## Transform the event traces
    norm_event_traces = [scaler.transform(x) for x in event_traces]

    # Perform PCA on mean traces
    pca_model = PCA(n_components=n_comps, svd_solver="full")
    ## Fit the model
    pca_model.fit(norm_mean_traces)
    ## Transform data
    avg_pop_vector = pca_model.transform(norm_mean_traces)
    event_pop_vector = [pca_model.transform(x) for x in norm_event_traces]

    # Assess trial-by-trial population similarity
    all_similarities, med_similarity = calc_pairwise_distances_btw_vectors(
        event_pop_vector
    )

    return avg_pop_vector, event_pop_vector, all_similarities, med_similarity


def simple_population_vector_analysis(
    dFoF,
    lever_active,
    activity_window=(0.5, 2),
    sampling_rate=60,
):
    """Function to analyze the trial to trial population vector correlation
    using simple correlations

    INPUT PARAMETERS
        dFoF - 2d np.array of the dFoF traces for each ROI, Columns represent
                each ROI.

        lever_active - 1d binary np.ndarray of the lever active trace

        activity_window - tuple specifying the time window around movements to
                            analyze in sec

        sampling_rate - int specifying the imaging rate

    OUTPUT PARAMETERS
        med_correlation - float of the median correlation

    """
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

    ind_traces, _ = d_utils.get_trace_mean_sem(
        dFoF,
        ROI_ids,
        timestamps,
        activity_window,
        sampling_rate,
    )

    ## Reorganize the traces
    ind_traces = list(ind_traces.values())
    ### Need to get traces for each event rather than for each roi
    event_traces = []
    for i, _ in enumerate(timestamps):
        temp_traces = [x[:, i] for x in ind_traces]
        ## Combine into one linear trace
        event_traces.append(np.concatenate(temp_traces))

    event_traces = np.array(event_traces)

    corr_length = event_traces.shape[1]
    # Perform the correlation
    med_correlation = correlate_btw_sessions(event_traces, event_traces, corr_length)

    return med_correlation
