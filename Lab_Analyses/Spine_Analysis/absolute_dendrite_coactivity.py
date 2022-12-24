import numpy as np
from Lab_Analyses.Spine_Analysis.spine_coactivity_utilities_v2 import (
    analyze_activity_trace,
    get_trace_coactivity_rates,
)
from Lab_Analyses.Spine_Analysis.spine_utilities import find_spine_classes
from Lab_Analyses.Utilities import activity_timestamps as tstamps


def absolute_dendrite_coactivity(
    spine_activity,
    spine_dFoF,
    spine_calcium,
    dend_activity,
    dend_dFoF,
    spine_groupings,
    spine_flags,
    constrain_matrix=None,
    activity_window=(-2, 4),
    sampling_rate=60,
    volume_norm=None,
):
    """Function to examine the absolute coactivity between spine and
        dendrties
        
        INPUT PARAMTERS
            spine_activity - 2d array of binarized spine activity traces
            
            spine_dFoF - 2d array of the spine dFoF traces
            
            spine_calcium - 2d array of the spine calcium traces
            
            dend_activity - 2d array of the binarized dendrtite activity
            
            dend_dFoF - 2d array of the dendrite dFoF traces
            
            spine_groupings - list of the spine groupings
            
            spine_flags - list containing spine flags
            
            spine_position - list or array of spine positions along dendrite
            
            constrain_matrix - 1d or 2d array to constrain activity matrices
            
            activity_window - tuple specifying the window you wish to analyze
            
            sampling_rate - int specifying the imaging rate
            
            volume_norm - tuple list of constants to normalize spine traces
            
    """

    # Sort out the spine groupings to make sure it is iterable
    if type(spine_groupings[0]) != list:
        spine_groupings = [spine_groupings]

    # Find eliminated spines
    el_spines = find_spine_classes(spine_flags, "Eliminated Spine")
    el_spines = np.array(el_spines)

    # Sort out normalization constants
    if volume_norm is not None:
        glu_norm_constants = volume_norm[0]
        ca_norm_constants = volume_norm[1]
    else:
        glu_norm_constants = np.array([None for x in range(spine_activity.shape[1])])
        ca_norm_constants = np.array([None for x in range(spine_activity.shape[1])])

    if constrain_matrix is not None:
        spine_activity = spine_activity * constrain_matrix

    center_point = int(np.absolute(activity_window[0] * sampling_rate))

    # Set up outputs
    coactivity_matrix = np.zeros(spine_activity.shape)
    coactivity_rate = np.zeros(spine_activity.shape[1])
    coactivity_rate_norm = np.zeros(spine_activity.shape[1])
    spine_fraction_coactive = np.zeros(spine_activity.shape[1])
    dend_fraction_coactive = np.zeros(spine_activity.shape[1])
    spine_coactive_amplitude = np.zeros(spine_activity.shape[1]) * np.nan
    spine_coactive_calcium = np.zeros(spine_activity.shape[1]) * np.nan
    spine_coactive_auc = np.zeros(spine_activity.shape[1]) * np.nan
    spine_coactive_calcium_auc = np.zeros(spine_activity.shape[1]) * np.nan
    dend_coactive_amplitude = np.zeros(spine_activity.shape[1]) * np.nan
    dend_coactive_auc = np.zeros(spine_activity.shape[1]) * np.nan
    relative_onset = np.zeros(spine_activity.shape[1]) * np.nan
    spine_coactive_traces = [None for i in range(spine_activity.shape[1])]
    spine_coactive_calcium_traces = [None for i in range(spine_activity.shape[1])]
    dend_coactive_traces = [None for i in range(spine_activity.shape[1])]

    # Iterate through each dendrite grouping
    for spines in spine_groupings:
        # Pull current spine grouping data
        s_activity = spine_activity[:, spines]
        s_dFoF = spine_dFoF[:, spines]
        s_calcium = spine_calcium[:, spines]
        d_activity = dend_activity[:, spines]
        d_dFoF = dend_dFoF[:, spines]
        curr_el_spines = el_spines[spines]
        curr_glu_constants = glu_norm_constants[spines]
        curr_ca_constants = ca_norm_constants[spines]

        # Analyze each spine individually
        for spine in range(s_activity.shape[1]):
            # Check if eliminated spine
            if curr_el_spines[spine]:
                continue
            # Get coactivity
            (
                c_rate,
                c_rate_norm,
                s_frac_active,
                d_frac_active,
                coactivity_trace,
            ) = get_trace_coactivity_rates(
                s_activity[:, spine], d_activity[:, spine], sampling_rate
            )

            if not np.sum(coactivity_trace):
                continue

            coactivity_matrix[:, spines[spine]] = coactivity_trace
            coactivity_rate[spines[spine]] = c_rate
            coactivity_rate_norm[spines[spine]] = c_rate_norm
            spine_fraction_coactive[spines[spine]] = s_frac_active
            dend_fraction_coactive[spines[spine]] = d_frac_active

            # Get local coactivity timestamps
            coactivity_stamps = tstamps.get_activity_timestamps(coactivity_trace)
            if len(coactivity_stamps) == 0:
                continue

            # Analyze activity traces when coactive
            ## Dendrite traces
            (d_traces, d_amp, d_auc, d_onset) = analyze_activity_trace(
                d_dFoF[:, spine],
                coactivity_stamps,
                activity_window=activity_window,
                center_onset=True,
                norm_constant=None,
                sampling_rate=sampling_rate,
            )
            dend_coactive_traces[spines[spine]] = d_traces
            dend_coactive_amplitude[spines[spine]] = d_amp
            dend_coactive_auc[spines[spine]] = d_auc

            ##Re-center timestamps around dendrite onsets
            corrected_stamps = tstamps.timestamp_onset_correction(
                coactivity_stamps, activity_window, d_onset, sampling_rate
            )
            (s_traces, s_amp, s_auc, s_onset) = analyze_activity_trace(
                s_dFoF[:, spine],
                corrected_stamps,
                activity_window=activity_window,
                center_onset=False,
                norm_constant=curr_glu_constants[spine],
                sampling_rate=sampling_rate,
            )
            (s_ca_traces, s_ca_amp, s_ca_auc, _) = analyze_activity_trace(
                s_calcium[:, spine],
                corrected_stamps,
                activity_window=activity_window,
                center_onset=False,
                norm_constant=curr_ca_constants[spine],
                sampling_rate=sampling_rate,
            )
            rel_onset = (int(s_onset) - center_point) / sampling_rate

            spine_coactive_traces[spines[spine]] = s_traces
            spine_coactive_amplitude[spines[spine]] = s_amp
            spine_coactive_auc[spines[spine]] = s_auc
            spine_coactive_calcium_traces[spines[spine]] = s_ca_traces
            spine_coactive_calcium[spines[spine]] = s_ca_amp
            spine_coactive_calcium_auc[spines[spine]] = s_ca_auc
            relative_onset[spines[spine]] = rel_onset

    return (
        coactivity_matrix,
        coactivity_rate,
        coactivity_rate_norm,
        spine_fraction_coactive,
        dend_fraction_coactive,
        spine_coactive_amplitude,
        spine_coactive_calcium,
        spine_coactive_auc,
        spine_coactive_calcium_auc,
        dend_coactive_amplitude,
        dend_coactive_auc,
        relative_onset,
        spine_coactive_traces,
        spine_coactive_calcium_traces,
        dend_coactive_traces,
    )
