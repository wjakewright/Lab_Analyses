import numpy as np

from Lab_Analyses.Spine_Analysis.spine_coactivity_utilities_v2 import (
    analyze_activity_trace,
    analyze_nearby_coactive_spines,
    get_trace_coactivity_rates,
)
from Lab_Analyses.Spine_Analysis.spine_utilities import find_spine_classes
from Lab_Analyses.Utilities import activity_timestamps as tstamps
from Lab_Analyses.Utilities.data_utilities import calculate_activity_event_rate
from Lab_Analyses.Utilities.quantify_movment_quality import quantify_movement_quality


def absolute_local_coactivity(
    mouse_id,
    spine_activity,
    spine_dFoF,
    spine_calcium,
    spine_groupings,
    spine_flags,
    spine_volumes,
    spine_positions,
    move_spines,
    lever_active,
    lever_force,
    partner_list=None,
    activity_window=(-2, 4),
    cluster_dist=5,
    sampling_rate=60,
    volume_norm=None,
):
    """Function to examine absolute local coactivity, or events when any other local spine
        is coactive with the target spine
        
        INPUT PARAMETERS
            spine_activity-  2d np.array of the binarrized spine activity. Columns=spines
            
            spine_dFoF - 2d np.array of the spine dFoF traces. Columns = spines
            
            spine_calcium - 2d np.array of spine calcium traces. Columns = spines
            
            spine_groupings - list of spine groupings
            
            spine_flags - list containing the spine flags
            
            spine_volumes - list or array of the estimated spine volumes (um)
            
            spine_positions - list or array of the spine positions along their parent dendrite
            
            move_spines - list of boolean containing movement spine idxs

            partner_list - boolean list specifying a subset of spines to analyze
                            coactivity rates for
            
            activity_window - tuple specifying the time window in sec over which to analyze
            
            cluster_dist - int specifying the distance in um tht is considered local
            
            sampling_rate - int specifying the imaging sampling rate
            
            volume_norm - tuple list of constants to normalize spine dFoF and calcium by
    """

    # Sort out the spine_groupings to make sure it is iterable
    if type(spine_groupings[0]) != list:
        spine_grouping = [spine_grouping]

    el_spines = find_spine_classes(spine_flags, "Eliminated Spine")
    el_spines = np.array(el_spines)

    if volume_norm is not None:
        glu_norm_constants = volume_norm[0]
        ca_norm_constants = volume_norm[1]
    else:
        glu_norm_constants = np.array([None for x in range(spine_activity.shape[1])])
        ca_norm_constants = np.array([None for x in range(spine_activity.shape[1])])

    if partner_list is None:
        partner_list = [True for x in range(spine_activity.shape[1])]

    # Set up outputs
    nearby_spine_idxs = [None for i in range(spine_activity.shape[1])]
    nearby_coactive_spine_idxs = [None for i in range(spine_activity.shape[1])]
    avg_nearby_spine_freq = np.zeros(spine_activity.shape[1]) * np.nan
    avg_nearby_coactive_spine_freq = np.zeros(spine_activity.shape[1]) * np.nan
    frac_nearby_MRSs = np.zeros(spine_activity.shape[1]) * np.nan
    nearby_coactive_spine_volumes = np.zeros(spine_activity.shape[1]) * np.nan
    local_coactivity_rate = np.zeros(spine_activity.shape[1])
    local_coactivity_rate_norm = np.zeros(spine_activity.shape[1])
    spine_fraction_coactive = np.zeros(spine_activity.shape[1])
    local_coactivity_matrix = np.zeros(spine_activity.shape)
    spine_coactive_amplitude = np.zeros(spine_activity.shape[1]) * np.nan
    spine_coactive_calcium = np.zeros(spine_activity.shape[1]) * np.nan
    spine_coactive_auc = np.zeros(spine_activity.shape[1]) * np.nan
    spine_coactive_calcium_auc = np.zeros(spine_activity.shape[1]) * np.nan
    spine_coactive_traces = [None for i in range(spine_activity.shape[1])]
    spine_coactive_calcium_traces = [None for i in range(spine_activity.shape[1])]
    spine_noncoactive_amplitude = np.zeros(spine_activity.shape[1]) * np.nan
    spine_noncoactive_calcium = np.zeros(spine_activity.shape[1]) * np.nan
    spine_noncoactive_auc = np.zeros(spine_activity.shape[1]) * np.nan
    spine_noncoactive_calcium_auc = np.zeros(spine_activity.shape[1]) * np.nan
    spine_noncoactive_traces = [None for i in range(spine_activity.shape[1])]
    spine_noncoactive_calcium_traces = [None for i in range(spine_activity.shape[1])]
    avg_coactive_spine_num = np.zeros(spine_activity.shape[1])
    sum_nearby_amplitude = np.zeros(spine_activity.shape[1]) * np.nan
    avg_nearby_amplitude = np.zeros(spine_activity.shape[1]) * np.nan
    sum_nearby_calcium = np.zeros(spine_activity.shape[1]) * np.nan
    avg_nearby_calcium = np.zeros(spine_activity.shape[1]) * np.nan
    sum_nearby_calcium_auc = np.zeros(spine_activity.shape[1]) * np.nan
    avg_nearby_calcium_auc = np.zeros(spine_activity.shape[1]) * np.nan
    avg_coactive_num_before = np.zeros(spine_activity.shape[1])
    sum_nearby_amplitude_before = np.zeros(spine_activity.shape[1]) * np.nan
    avg_nearby_amplitude_before = np.zeros(spine_activity.shape[1]) * np.nan
    sum_nearby_calcium_before = np.zeros(spine_activity.shape[1]) * np.nan
    avg_nearby_calcium_before = np.zeros(spine_activity.shape[1]) * np.nan
    avg_relative_nearby_onset = np.zeros(spine_activity.shape[1]) * np.nan
    sum_coactive_binary_traces = [None for i in range(spine_activity.shape[1])]
    sum_coactive_spine_traces = [None for i in range(spine_activity.shape[1])]
    avg_coactive_spine_traces = [None for i in range(spine_activity.shape[1])]
    sum_coactive_calcium_traces = [None for i in range(spine_activity.shape[1])]
    avg_coactive_calcium_traces = [None for i in range(spine_activity.shape[1])]
    avg_nearby_move_corr = np.zeros(spine_activity.shape[1]) * np.nan
    avg_nearby_move_relability = np.zeros(spine_activity.shape[1]) * np.nan
    avg_nearby_move_specificity = np.zeros(spine_activity.shape[1]) * np.nan

    # Iterate through each dendrite grouping
    for spines in spine_groupings:
        # Pull current spine grouping data
        s_dFoF = spine_dFoF[:, spines]
        s_activity = spine_activity[:, spines]
        s_calcium = spine_calcium[:, spines]
        curr_positions = spine_positions[spines]
        curr_el_spines = el_spines[spines]
        curr_volumes = spine_volumes[spines]
        curr_move_spines = move_spines[spines]
        curr_glu_norm_constants = glu_norm_constants[spines]
        curr_ca_norm_constants = ca_norm_constants[spines]
        curr_partner_spines = partner_list[spines]

        # Analyze each spine individually
        for spine in range(s_dFoF.shape[1]):
            # Get positional information and nearby spine idxs
            target_position = curr_positions[spine]
            relative_positions = np.array(curr_positions) - target_position
            relative_positions = np.absolute(relative_positions)
            nearby_spines = np.nonzero(relative_positions <= cluster_dist)[0]
            nearby_spines = [
                x for x in nearby_spines if not curr_el_spines[x] and x != spine
            ]
            nearby_spine_idxs[spine] = nearby_spines
            # Assess whether nearby spines display any coactivity
            nearby_coactive_spines = []
            for ns in nearby_spines:
                if np.sum(s_activity[:, spine] * s_activity[:, ns]):
                    nearby_coactive_spines.append(ns)
            # Refine nearby coactive spines based on partner list
            nearby_coactive_spines = [
                x for x in nearby_coactive_spines if curr_partner_spines[x] is True
            ]
            # Skip further analysis if no nearby coactive spines
            if len(nearby_coactive_spines) == 0:
                continue
            nearby_coactive_spines = np.array(nearby_coactive_spines)
            nearby_coactive_spine_idxs[spines[spine]] = spines[nearby_coactive_spines]

            # Get fraction of coactive spines that are MRSs
            nearby_move_spines = curr_move_spines[nearby_coactive_spines].astype(int)
            frac_nearby_MRSs[spines[spine]] = np.sum(nearby_move_spines) / len(
                nearby_move_spines
            )

            # Get average coactive spine volumes
            nearby_coactive_spine_volumes[spines[spine]] = np.nanmean(
                curr_volumes[nearby_coactive_spines]
            )

            # Get relative activity data
            curr_s_dFoF = s_dFoF[:, spine]
            curr_s_activity = s_activity[:, spine]
            curr_s_calcium = s_calcium[:, spine]
            nearby_s_dFoF = s_dFoF[:, nearby_coactive_spines]
            nearby_s_activity = s_activity[:, nearby_coactive_spines]
            nearby_s_calcium = s_calcium[:, nearby_coactive_spines]
            glu_constant = curr_glu_norm_constants[spine]
            ca_constant = curr_ca_norm_constants[spine]
            nearby_glu_constants = curr_glu_norm_constants[nearby_coactive_spines]
            nearby_ca_constants = curr_ca_norm_constants[nearby_coactive_spines]

            # Get activity freq of nearby spines
            nearby_freqs = []
            for i in nearby_spine_idxs:
                a = s_activity[:, i]
                freq = calculate_activity_event_rate(a, sampling_rate)
                nearby_freqs.append(freq)
            nearby_coactive_freqs = []
            for i in nearby_coactive_spine_idxs:
                a = s_activity[:, i]
                freq = calculate_activity_event_rate(a, sampling_rate)
                nearby_coactive_freqs.append(freq)
            avg_nearby_spine_freq.append(np.nanmean(nearby_freqs))
            avg_nearby_coactive_spine_freq.append(np.nanmean(nearby_coactive_freqs))

            # Get local coactivity trace, where at least one spine is coactive
            combined_nearby_activity = np.sum(nearby_s_activity, axis=1)
            combined_nearby_activity[combined_nearby_activity > 1] = 1

            # Get local coactivity rates
            (
                coactivity_rate,
                coactivity_rate_norm,
                spine_frac_active,
                _,
                coactivity_trace,
            ) = get_trace_coactivity_rates(
                curr_s_activity, combined_nearby_activity, sampling_rate
            )
            # Skip further analysis if no local coactivity for current spine
            if not np.sum(coactivity_trace):
                continue

            local_coactivity_rate[spines[spine]] = coactivity_rate
            local_coactivity_rate_norm[spines[spine]] = coactivity_rate_norm
            spine_fraction_coactive[spines[spine]] = spine_frac_active
            local_coactivity_matrix[:, spines[spine]] = coactivity_trace

            # Get local coactivity timestamps
            coactivity_stamps = tstamps.get_activity_timestamps(coactivity_trace)
            if len(coactivity_stamps) == 0:
                continue

            # Analyze activity traces when coactive
            ## Glutamate traces
            (s_traces, s_amp, s_auc, s_onset,) = analyze_activity_trace(
                curr_s_dFoF,
                coactivity_stamps,
                activity_window=activity_window,
                center_onset=True,
                norm_constant=glu_constant,
                sampling_rate=sampling_rate,
            )
            ## Calcium traces
            (s_ca_traces, s_ca_amp, s_ca_auc, _) = analyze_activity_trace(
                curr_s_calcium,
                coactivity_stamps,
                activity_window=activity_window,
                center_onset=True,
                norm_constant=ca_constant,
                sampling_rate=sampling_rate,
            )
            spine_coactive_amplitude[spines[spine]] = s_amp
            spine_coactive_calcium[spines[spine]] = s_ca_amp
            spine_coactive_auc[spines[spine]] = s_auc
            spine_coactive_calcium_auc[spines[spine]] = s_ca_auc
            spine_coactive_traces[spines[spine]] = s_traces
            spine_coactive_calcium_traces[spines[spine]] = s_ca_traces

            # Analyze activity traces when not coactive
            ## Get noncoactive trace
            noncoactive_trace = curr_s_activity - coactivity_trace
            noncoactive_trace[noncoactive_trace < 0] = 0
            noncoactive_stamps = tstamps.get_activity_timestamps(noncoactive_trace)
            ## skip if no isolated events
            if len(noncoactive_stamps) != 0:
                ## Glutamate traces
                (ns_traces, ns_amp, ns_auc, _) = analyze_activity_trace(
                    curr_s_dFoF,
                    noncoactive_stamps,
                    activity_window=activity_window,
                    center_onset=True,
                    norm_constant=glu_constant,
                    sampling_rate=sampling_rate,
                )
                ## Calcium traces
                (ns_ca_traces, ns_ca_amp, ns_ca_auc, _) = analyze_activity_trace(
                    curr_s_calcium,
                    noncoactive_stamps,
                    activity_window=activity_window,
                    center_onset=True,
                    norm_constant=ca_constant,
                    sampling_rate=sampling_rate,
                )
                spine_noncoactive_amplitude[spines[spine]] = ns_amp
                spine_noncoactive_calcium[spines[spine]] = ns_ca_amp
                spine_noncoactive_auc[spines[spine]] = ns_auc
                spine_noncoactive_calcium_auc[spines[spine]] = ns_ca_auc
                spine_noncoactive_traces[spines[spine]] = ns_traces
                spine_noncoactive_calcium_traces[spines[spine]] = ns_ca_traces

            # Analyze activity of nearby spines during coactivity
            ## Get only the onset stamps
            coactivity_stamps = [x[0] for x in coactivity_stamps]
            ## Center timestamps around activity onset
            corrected_stamps = tstamps.timestamp_onset_correction(
                coactivity_stamps, activity_window, s_onset, sampling_rate
            )

            (
                avg_coactive_s_num,
                sum_n_amplitude,
                avg_n_amplitude,
                sum_n_calcium,
                avg_n_calcium,
                sum_n_calcium_auc,
                avg_n_calcium_auc,
                avg_coactive_n_before,
                sum_n_amplitude_before,
                avg_n_amplitude_before,
                sum_n_calcium_before,
                avg_n_calcium_before,
                avg_n_onset,
                sum_coactive_b_traces,
                sum_coactive_s_traces,
                avg_coactive_s_traces,
                sum_coactive_ca_traces,
                avg_coactive_ca_traces,
            ) = analyze_nearby_coactive_spines(
                corrected_stamps,
                curr_s_activity,
                nearby_s_dFoF,
                nearby_s_calcium,
                nearby_s_activity,
                nearby_glu_constants,
                nearby_ca_constants,
                activity_window=activity_window,
                sampling_rate=sampling_rate,
            )

            avg_coactive_spine_num[spines[spine]] = avg_coactive_s_num
            sum_nearby_amplitude[spines[spine]] = sum_n_amplitude
            avg_nearby_amplitude[spines[spine]] = avg_n_amplitude
            sum_nearby_calcium[spines[spine]] = sum_n_calcium
            avg_nearby_calcium[spines[spine]] = avg_n_calcium
            sum_nearby_calcium_auc[spines[spine]] = sum_n_calcium_auc
            avg_nearby_calcium_auc[spines[spine]] = avg_n_calcium_auc
            avg_coactive_num_before[spines[spine]] = avg_coactive_n_before
            sum_nearby_amplitude_before[spines[spine]] = sum_n_amplitude_before
            avg_nearby_amplitude_before[spines[spine]] = avg_n_amplitude_before
            sum_nearby_calcium_before[spines[spine]] = sum_n_calcium_before
            avg_nearby_calcium_before[spines[spine]] = avg_n_calcium_before
            avg_relative_nearby_onset[spines[spine]] = avg_n_onset - s_onset
            sum_coactive_binary_traces[spines[spine]] = sum_coactive_b_traces
            sum_coactive_spine_traces[spines[spine]] = sum_coactive_s_traces
            avg_coactive_spine_traces[spines[spine]] = avg_coactive_s_traces
            sum_coactive_calcium_traces[spines[spine]] = sum_coactive_ca_traces
            avg_coactive_calcium_traces[spines[spine]] = avg_coactive_ca_traces

            # Analyze movement encoding of nearby coactive spines
            (
                _,
                _,
                _,
                move_corrs,
                move_frac_active,
                _,
                _,
                active_frac_move,
                _,
            ) = quantify_movement_quality(
                mouse_id,
                nearby_s_activity,
                lever_active,
                lever_force,
                threshold=0.5,
                sampling_rate=sampling_rate,
            )
            avg_nearby_move_corr[spines[spine]] = np.nanmean(move_corrs)
            avg_nearby_move_relability[spines[spine]] = np.nanmean(move_frac_active)
            avg_nearby_move_specificity[spines[spine]] = np.nanmean(active_frac_move)

    # Return final outputs
    return (
        nearby_spine_idxs,
        nearby_coactive_spine_idxs,
        avg_nearby_spine_freq,
        avg_nearby_coactive_spine_freq,
        frac_nearby_MRSs,
        nearby_coactive_spine_volumes,
        local_coactivity_rate,
        local_coactivity_rate_norm,
        spine_fraction_coactive,
        local_coactivity_matrix,
        spine_coactive_amplitude,
        spine_coactive_calcium,
        spine_coactive_auc,
        spine_coactive_calcium_auc,
        spine_coactive_traces,
        spine_coactive_calcium_traces,
        spine_noncoactive_amplitude,
        spine_noncoactive_calcium,
        spine_noncoactive_auc,
        spine_noncoactive_calcium_auc,
        spine_noncoactive_traces,
        spine_noncoactive_calcium_traces,
        avg_coactive_spine_num,
        sum_nearby_amplitude,
        avg_nearby_amplitude,
        sum_nearby_calcium,
        avg_nearby_calcium,
        sum_nearby_calcium_auc,
        avg_nearby_calcium_auc,
        avg_coactive_num_before,
        sum_nearby_amplitude_before,
        avg_nearby_amplitude_before,
        sum_nearby_calcium_before,
        avg_nearby_calcium_before,
        avg_relative_nearby_onset,
        sum_coactive_binary_traces,
        sum_coactive_spine_traces,
        avg_coactive_spine_traces,
        sum_coactive_calcium_traces,
        avg_coactive_calcium_traces,
        avg_nearby_move_corr,
        avg_nearby_move_relability,
        avg_nearby_move_specificity,
    )

